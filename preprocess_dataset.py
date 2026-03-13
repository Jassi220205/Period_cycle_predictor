import pandas as pd
import os

# Resolve project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

dataset_path = os.path.join(project_dir, "dataset", "menstrual_cycle.csv")
output_path = os.path.join(project_dir, "dataset", "cleaned_cycle_data.csv")

print("Loading dataset from:", dataset_path)

df = pd.read_csv(dataset_path)

print("\nInitial dataset shape:", df.shape)

# -----------------------------
# Convert target column
# -----------------------------
df["EstimatedDayofOvulation"] = pd.to_numeric(
    df["EstimatedDayofOvulation"], errors="coerce"
)

# -----------------------------
# Convert numeric feature columns
# -----------------------------
numeric_columns = [
    "LengthofLutealPhase",
    "TotalNumberofHighDays",
    "TotalNumberofPeakDays",
    "Age",
    "BMI"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Remove rows with missing critical values
# -----------------------------
required_columns = numeric_columns + ["EstimatedDayofOvulation"]

df = df.dropna(subset=required_columns)

print("\nDataset shape after cleaning:", df.shape)

# -----------------------------
# Select features for the model
# -----------------------------
features = [
    "CycleNumber",
    "LengthofCycle",
    "LengthofLutealPhase",
    "TotalNumberofHighDays",
    "TotalNumberofPeakDays",
    "Age",
    "BMI"
]

target = "EstimatedDayofOvulation"

model_df = df[features + [target]]

print("\nFinal dataset columns used for model:")
print(model_df.columns)

# -----------------------------
# Basic statistical summary
# -----------------------------
print("\nFeature statistics:\n")
print(model_df.describe())

# -----------------------------
# Save cleaned dataset
# -----------------------------
model_df.to_csv(output_path, index=False)

print("\nClean dataset saved at:", output_path)