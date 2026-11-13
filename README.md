# Customer Churn Prediction

An interactive machine learning application that predicts customer churn using historical customer data. The project uses Logistic Regression to identify customers who are likely to discontinue a service and provides insights through an interactive Streamlit dashboard.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave helps businesses take proactive measures to improve customer retention.

This project uses the IBM Telco Customer Churn dataset to develop a machine learning model that predicts customer churn based on customer demographics, account information, subscription details, and service usage.

The trained model is integrated into an interactive dashboard where users can explore customer data, analyse churn patterns, evaluate model performance, and predict churn for individual customers.

## Key Features

### 1. Dashboard Overview
- Displays important business metrics and customer statistics.
- Provides an overview of customer churn and retention.
- Highlights key customer insights through interactive visualisations.

### 2. Churn Analysis
- Analyses customer churn across different categories.
- Explores the relationship between customer attributes and churn.
- Provides interactive charts and filtering options.
- Helps identify patterns associated with customer cancellations.

### 3. Model Performance
- Evaluates the trained Logistic Regression model.
- Displays classification metrics.
- Visualises the confusion matrix.
- Presents feature coefficients to understand model behaviour.

### 4. Churn Prediction
- Allows users to enter individual customer details.
- Predicts the probability of customer churn.
- Classifies customers into Low, Medium, and High risk categories.
- Provides insights to support customer retention decisions.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualisation |
| Seaborn | Exploratory data analysis |
| Plotly | Interactive visualisations |
| Streamlit | Interactive web application |
| Joblib | Model saving and loading |

## Machine Learning Approach

**Algorithm:** Logistic Regression

The project follows a structured machine learning workflow:

1. Data Collection – Uses the IBM Telco Customer Churn dataset.
2. Data Preprocessing – Removes unnecessary columns, handles missing values, and converts data into suitable formats.
3. Exploratory Data Analysis – Examines customer characteristics and identifies churn patterns.
4. Feature Engineering – Separates numerical and categorical features.
5. Data Transformation – Applies StandardScaler to numerical features and OneHotEncoder to categorical features.
6. Model Training – Trains a Logistic Regression model using the processed dataset.
7. Model Evaluation – Evaluates performance using classification metrics and a confusion matrix.
8. Deployment – Integrates the trained model into a Streamlit dashboard for interactive predictions.

## Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains customer information from a telecommunications company, including:

- Customer demographics
- Gender and senior citizen status
- Tenure and contract details
- Internet and phone services
- Monthly and total charges
- Customer churn status

**Dataset size:** 7,043 customer records and 21 columns.

The dataset is used to understand customer behaviour and train the machine learning model.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│
├── outputs/
│   ├── churn_model.pkl
│   └── cleaned_churn_data.csv
│
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Execution

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone <your-github-repository-link>
```

### 2. Navigate to the Project Folder

```bash
cd Customer-Churn-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Model and Output Files

The project includes the following pre-generated files:

- `outputs/cleaned_churn_data.csv` – Processed customer dataset used by the application.
- `outputs/churn_model.pkl` – Trained Logistic Regression model used for predictions.

Both files must be available in the repository for the deployed application to work.

## Project Objective

The primary objective of this project is to apply machine learning techniques to a real-world business problem by predicting customer churn and identifying customers who may be at risk of leaving.

It demonstrates the practical application of data preprocessing, exploratory data analysis, machine learning, model evaluation, and interactive dashboard development.

## Author

**Yeshika**

B.Tech – Artificial Intelligence and Data Science

GitHub: yeshikadr