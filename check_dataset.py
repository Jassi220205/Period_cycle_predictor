import pandas as pd
import os

# Get location of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go to project root
project_dir = os.path.dirname(script_dir)

# Construct dataset path
dataset_path = os.path.join(project_dir, "dataset", "menstrual_cycle.csv")

print("Looking for dataset at:", dataset_path)

df = pd.read_csv(dataset_path)

print(df.head())
print(df.info())