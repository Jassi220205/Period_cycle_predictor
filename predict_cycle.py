import os
import joblib
import pandas as pd
from datetime import datetime, timedelta


# --------------------------------------------------
# Resolve paths
# --------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)

model_path = os.path.join(project_dir, "model", "cycle_prediction_model.pkl")
scaler_path = os.path.join(project_dir, "model", "scaler.pkl")


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

last_period_date = input("Enter last period start date (YYYY-MM-DD): ")

cycle_number = int(input("Cycle number: "))
cycle_length = int(input("Average cycle length: "))
luteal_phase = int(input("Luteal phase length: "))
high_days = int(input("Total high fertility days: "))
peak_days = int(input("Total peak fertility days: "))
age = int(input("Age: "))
bmi = float(input("BMI: "))


# --------------------------------------------------
# Convert last period date
# --------------------------------------------------

last_period_date = datetime.strptime(last_period_date, "%Y-%m-%d")


# --------------------------------------------------
# Prepare feature vector
# --------------------------------------------------

features = pd.DataFrame([[
    cycle_number,
    cycle_length,
    luteal_phase,
    high_days,
    peak_days,
    age,
    bmi
]], columns=[
    "CycleNumber",
    "LengthofCycle",
    "LengthofLutealPhase",
    "TotalNumberofHighDays",
    "TotalNumberofPeakDays",
    "Age",
    "BMI"
])


# --------------------------------------------------
# Scale features
# --------------------------------------------------

scaled_features = scaler.transform(features)


# --------------------------------------------------
# Predict ovulation day
# --------------------------------------------------

predicted_ovulation_day = model.predict(scaled_features)[0]

predicted_ovulation_day = round(predicted_ovulation_day)


# --------------------------------------------------
# Calculate ovulation date
# --------------------------------------------------

ovulation_date = last_period_date + timedelta(days=predicted_ovulation_day)


# --------------------------------------------------
# Calculate next period
# --------------------------------------------------

next_period_date = ovulation_date + timedelta(days=luteal_phase)


# --------------------------------------------------
# Fertile window
# --------------------------------------------------

fertile_start = ovulation_date - timedelta(days=5)
fertile_end = ovulation_date + timedelta(days=1)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n----- Prediction Results -----")

print("Predicted Ovulation Day:", predicted_ovulation_day)

print("Predicted Ovulation Date:", ovulation_date.date())

print("Predicted Next Period Date:", next_period_date.date())

print("Fertile Window:", fertile_start.date(), "to", fertile_end.date())

print("------------------------------")