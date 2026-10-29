

import os
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_DIR = "data"
FIG_DIR = "figures"
SHOW_PLOTS = True          # False = only save the figures, do not open windows
TARGET = "remaining_useful_cycles"

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)
plt.rcParams["figure.figsize"] = (7, 4)
os.makedirs(FIG_DIR, exist_ok=True)


def title(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def save_show(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------------
# 1. Load and merge the data
# ----------------------------------------------------------------------------
title("1. Load and merge the data")

if not os.path.exists(os.path.join(DATA_DIR, "battery_observations.csv")):
    with zipfile.ZipFile("archive.zip") as z:
        z.extractall(DATA_DIR)

obs = pd.read_csv(os.path.join(DATA_DIR, "battery_observations.csv"))
meta = pd.read_csv(os.path.join(DATA_DIR, "battery_metadata.csv"))
df = obs.merge(meta, on="battery_id", how="left")

print("observations:", obs.shape, "| batteries:", meta.shape[0])
print(df["dataset_split"].value_counts())

# ----------------------------------------------------------------------------
# 2. Data understanding
# ----------------------------------------------------------------------------
title("2. Data understanding")

# 2.1 missing values
miss = df.isna().sum()
print("Missing values:\n", miss[miss > 0])

# 2.2 batteries that never reached end of life have no RUL (censored)
has_rul = df.groupby("battery_id")[TARGET].apply(lambda s: s.notna().any())
print("\nbatteries with RUL:", has_rul.sum(), "| without RUL:", (~has_rul).sum())

# 2.3 eol_reached is constant inside a battery -> it does NOT mean "this cycle is the end of life"
const = (df.groupby("battery_id")["eol_reached"].nunique() == 1).all()
print("eol_reached constant per battery:", const)

# 2.4 RUL and state of health for one battery
b = df[df["battery_id"] == "BAT-00001"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(b["cycle_index"], b[TARGET])
ax[0].set_title("RUL vs cycle (BAT-00001)")
ax[1].plot(b["cycle_index"], b["state_of_health_pct"])
ax[1].set_title("SOH vs cycle (BAT-00001)")
for a in ax:
    a.set_xlabel("cycle index")
save_show("01_one_battery.png")

# ----------------------------------------------------------------------------
# 3. Cleaning
# ----------------------------------------------------------------------------
# Decisions (to explain in the report):
#  - remove the batteries with no RUL (censored, target unknown)
#  - pack_temp_c has a few missing values: fill inside each battery (previous/next
#    cycle), then with the train median if a gap remains
#  - excluded on purpose: remaining_useful_efc (same quantity as the target in other
#    units), degradation_status (derived from health), eol_reached (misleading flag)
title("3. Cleaning")

df = df[df["battery_id"].isin(has_rul[has_rul].index)].copy()
df = df.sort_values(["battery_id", "cycle_index"])

df["pack_temp_c"] = df.groupby("battery_id")["pack_temp_c"].transform(lambda s: s.ffill().bfill())
train_med = df.loc[df["dataset_split"] == "Train", "pack_temp_c"].median()
df["pack_temp_c"] = df["pack_temp_c"].fillna(train_med)

print("rows kept:", len(df), "| batteries:", df["battery_id"].nunique())
print("missing in target / pack_temp_c:", df[TARGET].isna().sum(), df["pack_temp_c"].isna().sum())

# ----------------------------------------------------------------------------
# 4. Variables
# ----------------------------------------------------------------------------
# 6 explanatory variables (rubric: 3 to 8). state_of_health_pct is the strongest one,
# so we also run a version WITHOUT it (section 8).
title("4. Variables")

FEATURES = [
    "cycle_index",
    "state_of_health_pct",
    "internal_resistance_mOhm",
    "pack_temp_c",
    "discharge_c_rate",
    "depth_of_discharge_pct",
]
FEATURES_NO_SOH = [f for f in FEATURES if f != "state_of_health_pct"]

print(df[FEATURES + [TARGET]].describe().round(2))

corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
print("\nCorrelation with the target:\n", corr.round(2))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].hist(df[TARGET], bins=50)
ax[0].set_title("Distribution of RUL")
ax[0].set_xlabel("remaining useful cycles")
corr.plot.barh(ax=ax[1])
ax[1].set_title("Correlation with RUL")
save_show("02_target_and_correlation.png")

# ----------------------------------------------------------------------------
# 5. Train / validation / test split
# ----------------------------------------------------------------------------
# We use the split PROVIDED with the dataset, by battery. Cycles of the same battery are
# very similar: a random row split would put near-copies in train and test and inflate
# the score.
title("5. Train / validation / test split")

train = df[df["dataset_split"] == "Train"]
val = df[df["dataset_split"] == "Validation"]
test = df[df["dataset_split"] == "Test"]

for name, part in [("train", train), ("validation", val), ("test", test)]:
    print(f"{name:10s} rows={len(part):6d} batteries={part['battery_id'].nunique()}")

# safety check: no battery in two sets
assert not set(train.battery_id) & set(test.battery_id)
assert not set(train.battery_id) & set(val.battery_id)

# ----------------------------------------------------------------------------
# 6. Models
# ----------------------------------------------------------------------------
title("6. Models")


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def fit_eval(model, feats, fit_on, eval_on):
    model.fit(fit_on[feats], fit_on[TARGET])
    pred = model.predict(eval_on[feats])
    return evaluate(eval_on[TARGET], pred), pred


# naive baseline: always predict the mean RUL of the training set
base_val = evaluate(val[TARGET], np.full(len(val), train[TARGET].mean()))
print("Baseline (mean) on validation:", {k: round(float(v), 3) for k, v in base_val.items()})

# 6.1 tune hyper-parameters on the VALIDATION set (never on the test set)
depths = list(range(2, 16))
ks = [3, 5, 10, 20, 50]
tree_val = [
    fit_eval(DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE), FEATURES, train, val)[0]["MAE"]
    for d in depths
]
knn_val = [
    fit_eval(make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k)), FEATURES, train, val)[0]["MAE"]
    for k in ks
]
best_depth = depths[int(np.argmin(tree_val))]
best_k = ks[int(np.argmin(knn_val))]
print("best tree depth:", best_depth, "| best k:", best_k)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(depths, tree_val, marker="o")
ax[0].set_title("Tree: validation MAE vs depth")
ax[0].set_xlabel("max_depth")
ax[1].plot(ks, knn_val, marker="o")
ax[1].set_title("kNN: validation MAE vs k")
ax[1].set_xlabel("k")
for a in ax:
    a.set_ylabel("MAE (cycles)")
save_show("03_tuning.png")

# 6.2 final models: fit on train + validation, evaluate ONCE on the test set
models = {
    "Linear regression": LinearRegression(),
    "Decision tree": DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE),
    "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=best_k)),
}

trainval = pd.concat([train, val])
results, preds = {}, {}
results["Baseline (mean)"] = evaluate(test[TARGET], np.full(len(test), trainval[TARGET].mean()))
for name, m in models.items():
    results[name], preds[name] = fit_eval(m, FEATURES, trainval, test)

print("\nTest set results:")
print(pd.DataFrame(results).T.round(3))

# ----------------------------------------------------------------------------
# 7. Visualisation of the results
# ----------------------------------------------------------------------------
title("7. Visualisation of the results")

# 7.1 actual vs predicted (random sample of test rows so the plot stays readable)
rng = np.random.RandomState(RANDOM_STATE)
idx = rng.choice(len(test), size=3000, replace=False)
y = test[TARGET].values

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
for a, (name, p) in zip(ax, preds.items()):
    a.scatter(y[idx], p[idx], s=4, alpha=0.3)
    a.plot([0, y.max()], [0, y.max()], "r--", lw=1)
    a.set_title(f"{name}  (R2={results[name]['R2']:.2f})")
    a.set_xlabel("actual RUL")
ax[0].set_ylabel("predicted RUL")
save_show("04_actual_vs_predicted.png")

# 7.2 residuals
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, p in preds.items():
    ax[0].hist(y - p, bins=60, alpha=0.5, label=name)
ax[0].set_title("Residuals (actual - predicted)")
ax[0].set_xlabel("cycles")
ax[0].legend()

p = preds["Decision tree"]
ax[1].scatter(y[idx], (y - p)[idx], s=4, alpha=0.3)
ax[1].axhline(0, color="r", lw=1)
ax[1].set_title("Decision tree: residual vs actual RUL")
ax[1].set_xlabel("actual RUL")
ax[1].set_ylabel("residual")
save_show("05_residuals.png")

# 7.3 interpretation: linear coefficients (standardised) and tree importances
lin_std = make_pipeline(StandardScaler(), LinearRegression()).fit(trainval[FEATURES], trainval[TARGET])
coefs = pd.Series(lin_std[-1].coef_, index=FEATURES).sort_values()

tree = DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE).fit(trainval[FEATURES], trainval[TARGET])
imp = pd.Series(tree.feature_importances_, index=FEATURES).sort_values()

print("Standardised linear coefficients:\n", coefs.round(1))
print("\nTree feature importances:\n", imp.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
coefs.plot.barh(ax=ax[0])
ax[0].set_title("Linear regression: standardised coefficients")
imp.plot.barh(ax=ax[1])
ax[1].set_title("Decision tree: feature importance")
save_show("06_interpretation.png")

# ----------------------------------------------------------------------------
# 8. Robustness checks
# ----------------------------------------------------------------------------
title("8. Robustness checks")

# 8.1 without state of health (harder, closer to a case where SOH is not measured)
rows = []
for name, m in models.items():
    r, _ = fit_eval(m, FEATURES_NO_SOH, trainval, test)
    rows.append({"model": name, **r, "variables": "without SOH"})
    rows.append({"model": name, **results[name], "variables": "with SOH"})
print("8.1 With vs without state of health (test set):")
print(pd.DataFrame(rows).round(3).sort_values(["model", "variables"]).to_string(index=False))

# 8.2 cross-validation grouped by battery (on train + validation)
gkf = GroupKFold(n_splits=5)
cv_rows = []
for name, m in models.items():
    scores = -cross_val_score(
        m, trainval[FEATURES], trainval[TARGET],
        groups=trainval["battery_id"], cv=gkf, scoring="neg_mean_absolute_error",
    )
    cv_rows.append({"model": name, "MAE mean": scores.mean(), "MAE std": scores.std()})
print("\n8.2 5-fold cross-validation grouped by battery (MAE):")
print(pd.DataFrame(cv_rows).round(2).to_string(index=False))

# 8.3 error per battery (does the model work for every battery?)
t = test.copy()
t["pred"] = preds["Decision tree"]
per_bat = pd.Series(
    {bid: mean_absolute_error(g[TARGET], g["pred"]) for bid, g in t.groupby("battery_id")}
)
print("\n8.3 Decision tree MAE per test battery:")
print(per_bat.describe().round(1))

per_bat.sort_values().plot.bar(figsize=(11, 3.5), title="Decision tree: MAE per test battery")
plt.ylabel("MAE (cycles)")
save_show("07_error_per_battery.png")

# 8.4 error by chemistry / climate zone (metadata is NOT a model input, only used to analyse errors)
t["abs_err"] = (t[TARGET] - t["pred"]).abs()
print("\nMean absolute error by chemistry:\n", t.groupby("chemistry")["abs_err"].mean().round(1))
print("\nMean absolute error by climate zone:\n", t.groupby("climate_zone")["abs_err"].mean().round(1))

# ----------------------------------------------------------------------------
# 9. Interpretation and limits (write this part in your own words in the report)
# ----------------------------------------------------------------------------
# 1. Which model is best on the test set, and by how much versus the mean baseline?
# 2. Why does the tree beat linear regression? (actual vs predicted plot, residuals)
# 3. What happens without state of health? What does it say about a real device?
# 4. Are errors larger at the start or at the end of life? Which one matters more?
# 5. Limits: batteries without RUL removed, data may be simulated, cycles of one
#    battery are not independent, one target definition (SOH = 70%).
title("Done - figures saved in ./" + FIG_DIR)
