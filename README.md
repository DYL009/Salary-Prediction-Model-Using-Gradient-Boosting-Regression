# Salary Prediction Model Using Gradient Boosting Regression

## Overview
This project predicts employee salaries based on various features like job role, experience level, company location, and more. The model is built using a Gradient Boosting Regressor pipeline, incorporating preprocessing steps like encoding and scaling. It achieves consistent performance across cross-validation and evaluation metrics.

---

## Features
The dataset contains the following columns:
- `work_year`: The year of the observation.
- `job_category`: Category of the job role.
- `salary_currency`: The currency of the provided salary.
- `salary`: The salary amount in the given currency.
- `salary_in_usd`: The salary amount converted to USD.
- `employee_residence`: The country of residence of the employee.
- `experience_level`: The level of experience (`Junior`, `Mid`, `Senior`, etc.).
- `employment_type`: Full-time, part-time, etc.
- `work_setting`: Remote, hybrid, or on-site.
- `company_location`: The company's country.
- `company_size`: The size of the company (e.g., `Small`, `Medium`, `Large`).
- And other boolean job-related columns.

---

## Model Workflow
1. **Data Preprocessing**:
   - Standardized text columns to lowercase.
   - Encoded categorical features using one-hot encoding.
   - Scaled numerical features using StandardScaler.
   - Combined preprocessing steps using ColumnTransformer.

2. **Model**:
   - Gradient Boosting Regressor is used for regression tasks.
   - Model trained on 80% of the data and tested on 20%.

3. **Evaluation**:
   - Metrics:
     - Mean Absolute Error (MAE): Measures average absolute prediction error.
     - Root Mean Squared Error (RMSE): Penalizes larger errors more heavily.
     - R-squared (R²): Measures how well the model explains variance in the target variable.
   - Cross-validation ensures robust performance.

---

## Results
- **MAE**: 42,671.51
- **RMSE**: 49,601.34
- **R²**: -0.02

**Cross-Validation Metrics**:
- Mean MAE: 42,674.87 ± 693.06
- Mean RMSE: 49,550.78 ± 691.59
- Mean R²: -0.0268 ± 0.0034

---

## How to Use
### Prerequisites
- Python 3.8 or higher
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`


