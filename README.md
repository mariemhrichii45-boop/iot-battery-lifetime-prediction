# 🔋 IoT Battery Lifetime Prediction Using Machine Learning

## 📌 Overview

This project aims to predict the remaining battery lifetime of IoT devices using Machine Learning regression algorithms.

Battery autonomy is an important factor in IoT systems, especially for devices operating remotely or with limited access to charging infrastructure. Predicting the remaining operating time can help optimize energy consumption, improve maintenance planning, and increase device reliability.

The project uses historical battery and operating-condition data to train and evaluate different Machine Learning models capable of estimating the remaining battery lifetime.

## 🎯 Objectives

- Analyze battery-related datasets and identify relevant features.
- Explore the relationship between operating conditions and battery consumption.
- Preprocess and prepare data for Machine Learning.
- Develop and compare multiple regression models.
- Predict the remaining battery lifetime of IoT devices.
- Evaluate model performance using appropriate regression metrics.

## 📊 Dataset

**Dataset source:** Kaggle

🔗 **Kaggle Dataset:** [https://www.kaggle.com/datasets/razanihababdellatif/battery-remaining-useful-life-rul-dataset/code]

The dataset will be used to train and evaluate the proposed regression models.

### Expected Features

The following features are considered relevant to battery lifetime prediction, depending on their availability in the selected dataset:

| Feature | Description |
|---|---|
| Battery Level | Current battery charge level |
| Current Consumption | Electrical current consumed by the device |
| Operating Temperature | Device operating temperature |
| Transmission Frequency | Frequency of wireless data transmissions |
| Uptime | Device operating time |

**Target Variable:** Remaining Battery Lifetime (hours).

> Note: The final features and target variable will depend on the actual structure and documentation of the selected Kaggle dataset.

## 🤖 Machine Learning Models

The project will investigate and compare several regression algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors (KNN) Regressor

The models will be trained and evaluated using a consistent dataset split and evaluation methodology.

## 📈 Evaluation Metrics

Model performance will be assessed using:

| Metric | Purpose |
|---|---|
| MAE | Measures the average absolute prediction error |
| RMSE | Penalizes larger prediction errors |
| R² Score | Measures how well the model explains target variability |

## 🛠️ Technologies & Tools

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn




## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mariemhrichii45-boop/iot-battery-lifetime-prediction.git

cd iot-battery-lifetime-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Project Workflow

1. Dataset acquisition and exploration.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Feature selection and engineering.
5. Model training.
6. Model comparison and evaluation.
7. Remaining battery lifetime prediction.

## 📊 Results

Model performance and comparison results will be added after training and evaluation.

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD |

## 🔮 Future Improvements

- Develop an interactive prediction dashboard using Streamlit.
- Investigate feature importance and model explainability.
- Optimize hyperparameters to improve prediction accuracy.
- Explore real-time battery monitoring using ESP32.
- Integrate the trained model into an IoT monitoring system.

## 👨‍💻 Author

**Maryem Hrichi**

Computer Engineering Student – IoT & Embedded Systems

Faculty of Sciences of Tunis

[GitHub](https://github.com/mariemhrichii45-boop) | [LinkedIn](https://www.linkedin.com/in/mariem-hrichi-77a588264/)

---

*This project was developed as part of an academic Machine Learning project.*