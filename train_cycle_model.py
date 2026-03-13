import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib


# ---------------------------------------------------
# Resolve project paths
# ---------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

dataset_path = os.path.join(project_dir, "dataset", "cleaned_cycle_data.csv")

model_dir = os.path.join(project_dir, "model")

os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "cycle_prediction_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")


# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nPreview:")
print(df.head())


# ---------------------------------------------------
# Define features and target
# ---------------------------------------------------

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

X = df[features]

y = df[target]


# ---------------------------------------------------
# Train-test split
# ---------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ---------------------------------------------------
# Feature scaling
# ---------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------
# Initialize model
# ---------------------------------------------------

print("\nInitializing Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------
# Train model
# ---------------------------------------------------

print("\nTraining model...")

model.fit(X_train_scaled, y_train)

print("Model training completed.")


# ---------------------------------------------------
# Make predictions
# ---------------------------------------------------

print("\nMaking predictions...")

predictions = model.predict(X_test_scaled)


# ---------------------------------------------------
# Evaluate model
# ---------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

print("\nModel Evaluation")

print("Mean Absolute Error (MAE):", round(mae, 3))

print("Mean Squared Error (MSE):", round(mse, 3))


# ---------------------------------------------------
# Save model and scaler
# ---------------------------------------------------

print("\nSaving trained model...")

joblib.dump(model, model_path)

joblib.dump(scaler, scaler_path)

print("Model saved at:", model_path)
print("Scaler saved at:", scaler_path)

print("\nTraining pipeline completed successfully.")