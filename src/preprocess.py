import pandas as pd
from pathlib import Path

project_path = Path(__file__).resolve().parent.parent

input_path = project_path / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
output_path = project_path / "outputs" / "cleaned_churn_data.csv"

df = pd.read_csv(input_path)

print("Original Dataset Shape:", df.shape)

df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df.reset_index(drop=True, inplace=True)

df.to_csv(output_path, index=False)

print("\nDATA CLEANING COMPLETED!")
print("Cleaned Dataset Shape:", df.shape)
print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nCleaned dataset saved successfully!")