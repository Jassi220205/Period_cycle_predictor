Menstrual Cycle Prediction System (ML-Based)

A hybrid Machine Learning + Rule-Based system designed to predict: -
Ovulation Date - Next Period Date - Fertile Window

Problem Statement

-   Predict ovulation day
-   Estimate next menstrual period
-   Identify fertile window

System Architecture

Raw Dataset → Preprocessing → Feature Selection → ML Model → Ovulation
Prediction → Rule Engine → Final Output

Tech Stack

Python, Pandas, NumPy, Scikit-learn, Random Forest, StandardScaler,
Joblib

Dataset

1665 records, 80 features reduced to 7 key features

Data Preprocessing

-   Cleaning
-   Feature selection
-   Scaling

Model Training

-   Train-test split (80/20)
-   MAE ≈ 1–2 days

Prediction Logic

Next Period = Ovulation + Luteal Phase Fertile Window = Ovulation - 5 to
+1 days

Project Structure

dataset/, model/, scripts/

Key Insights

Hybrid ML + rules improves reliability

Limitations

-   No time-series modeling
-   Limited data

Conclusion

Accurate + interpretable healthcare ML system.
