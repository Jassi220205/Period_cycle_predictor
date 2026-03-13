#Menstrual Cycle Prediction System
A machine learning system that predicts ovulation date, next period date, and fertile window using menstrual cycle data and biological rules.

This project demonstrates how machine learning combined with reproductive biology can estimate menstrual cycle events.

Project Overview
The system predicts menstrual cycle events in two stages:

Machine Learning Model

Predicts the ovulation day based on cycle features.

Biological Rule Engine

Uses ovulation prediction to estimate:

next period date

fertile window

The ML model used is a Random Forest Regressor trained on menstrual cycle data.

System Architecture
Pipeline:


Menstrual Cycle Dataset
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Random Forest Model
        ↓
Ovulation Prediction
        ↓
Biological Rule Engine
        ↓
Final Predictions
Final outputs:

Predicted Ovulation Date

Predicted Next Period Date

Fertile Window

Project Structure

menstrual-cycle-predictor/

dataset/
    menstrual_cycle.csv
    cleaned_cycle_data.csv

model/
    cycle_prediction_model.pkl
    scaler.pkl

scripts/
    check_dataset.py
    preprocess_dataset.py
    train_cycle_model.py
    predict_cycle.py
Dataset
The dataset contains menstrual cycle records with biological variables such as:

CycleNumber

LengthofCycle

LengthofLutealPhase

TotalNumberofHighDays

TotalNumberofPeakDays

Age

BMI

EstimatedDayofOvulation

After preprocessing, irrelevant columns were removed and the dataset was cleaned for model training.

Machine Learning Model
Model used:


RandomForestRegressor
Reasons:

Handles nonlinear relationships

Works well with structured tabular data

Reduces overfitting using ensemble learning

Evaluation metrics:


Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Typical performance:


MAE ≈ 1–2 days
Installation
Clone the repository:


git clone https://github.com/yourusername/menstrual-cycle-predictor.git
cd menstrual-cycle-predictor
Install dependencies:


pip install pandas scikit-learn joblib
Usage
1. Preprocess the Dataset

python scripts/preprocess_dataset.py
This cleans the raw dataset and creates:


dataset/cleaned_cycle_data.csv
2. Train the Model

python scripts/train_cycle_model.py
This trains the Random Forest model and saves:


model/cycle_prediction_model.pkl
model/scaler.pkl
3. Predict Cycle Events
Run:


python scripts/predict_cycle.py
Example input:


Enter last period start date: 2026-02-26
Cycle number: 4
Average cycle length: 27
Luteal phase length: 14
Total high fertility days: 4
Total peak fertility days: 1
Age: 21
BMI: 21.5
Example output:


Predicted Ovulation Day: 13
Predicted Ovulation Date: 2026-03-11
Predicted Next Period Date: 2026-03-25
Fertile Window: 2026-03-06 to 2026-03-12
Biological Assumptions Used
Ovulation separates the menstrual cycle into two phases:


Follicular Phase
Ovulation
Luteal Phase
The next period is estimated using:


Next Period = Ovulation Date + Luteal Phase Length
Fertility window:


Ovulation − 5 days → Ovulation + 1 day
This is based on sperm survival time in reproductive biology.

Technologies Used
Python

Pandas

Scikit-learn

Joblib

Applications
This system demonstrates potential applications in:

menstrual health tracking

fertility awareness tools

reproductive health analytics

personalized health AI systems

Future Improvements
Possible enhancements include:

time-series models (LSTM)

personalized user modeling

irregular cycle detection

confidence scoring

mobile or web interface

License
This project is intended for educational and research purposes.
