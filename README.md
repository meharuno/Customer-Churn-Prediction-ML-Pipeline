# 🧠 Customer Churn Prediction ML Pipeline

> Predict whether a customer will churn (i.e., stop using a service) based on behavioral and account-related features using a full machine learning pipeline.

---

## 📌 Project Overview

Customer churn is a major concern for subscription-based businesses. In this project, we build a machine learning pipeline to:

- Clean and preprocess customer data.
- Perform exploratory data analysis (EDA).
- Train and evaluate several classification models.
- Select the best-performing model.
- Gain insights into the most influential factors driving churn.

---

## 🗂️ Dataset

- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Format**: CSV
- **Rows**: 7043 customers
- **Target Variable**: `Churn` (Yes/No)

---

## 🔧 Pipeline Components

### 1. Data Preprocessing
- Handling missing values using `SimpleImputer`, `KNNImputer`, and `IterativeImputer`.
- Encoding categorical features using `LabelEncoder` and `OneHotEncoder`.
- Scaling features with `StandardScaler` and `MinMaxScaler`.

### 2. Exploratory Data Analysis (EDA)
- Univariate and bivariate analysis using `seaborn`, `matplotlib`, and `plotly`.
- Distribution plots, correlation heatmaps, churn rate by features.

### 3. Model Building
Tested multiple classifiers:
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- XGBoost
- Naive Bayes
- AdaBoost
- Gradient Boosting

### 4. Model Evaluation
- Metrics:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)
  - Cross-validation
- Feature importance analysis

---

## 🏆 Best Model

The best-performing model was **XGBoost (XGBClassifier)**, selected after a comprehensive hyperparameter tuning process using `RandomizedSearchCV`.

### ✅ Why XGBoost?
- It consistently outperformed other models (e.g., Logistic Regression, Random Forest, Gradient Boosting) in terms of **accuracy** on the test set.
- It is robust to overfitting and highly flexible.
- Offers excellent feature importance insights.

### 🔧 Final Pipeline
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(...))  # tuned hyperparameters
])
```
### 📈 Metrics Achieved
- Accuracy: 0.69

- Other Metrics: Precision, Recall, F1-Score are available in the classification report.

---
### 🛠️ Tools & Libraries
Python

- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- XGBoost
- Jupyter Notebook
  
---
### 📁 Folder Structure
📦 customer-churn-prediction-ml-pipeline
├── customer-churn-prediction-ml-pipeline.ipynb
├── README.md
├── churn_data.csv
└── models/
    └── (optional) saved_model.pkl

---
### 🚀 Future Improvements
- Hyperparameter tuning with Optuna or Bayesian Optimization.
- Add model explainability tools (e.g., SHAP, LIME).
- Deploy model via FastAPI or Flask.
- Create a live dashboard using Streamlit or Dash.

---
### 📬 Contact
For questions or collaboration, feel free to reach out!










