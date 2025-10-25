import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Define Paths Robustly ---
# Get the directory where this script (1_eda.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

# Plots will be saved in a folder *inside* the current 'eda' directory
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'eda_plots') # Save plots relative to this script
os.makedirs(PLOTS_DIR, exist_ok=True)
# --- End of Path Fix ---

# Load data
try:
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
except FileNotFoundError:
    print(f"Error: Could not find data files in {DATA_DIR}")
    print("Please make sure train.csv and test.csv are in the 'data' folder.")
    sys.exit(1)

print("--- Train Data Info ---")
df_train.info()

print("\n--- Test Data Info ---")
df_test.info()

print("\n--- Train Data Statistical Summary ---")
print(df_train.describe())

print("\n--- Train Data Missing Values ---")
print(df_train.isnull().sum())

# --- Target Variable Analysis (Transport_Cost) ---
negative_costs = df_train[df_train['Transport_Cost'] <= 0]
print(f"\nFound {len(negative_costs)} rows with non-positive Transport_Cost.")

# Plot 1: Distribution of Transport_Cost
plt.figure(figsize=(12, 6))
sns.histplot(df_train['Transport_Cost'], kde=True, bins=100)
plt.title('Distribution of Transport_Cost (Original)')
plt.xlabel('Transport_Cost')
plt.ylabel('Frequency')
plt.savefig(os.path.join(PLOTS_DIR, '1_target_distribution_original.png'))
print(f"Saved plot: {os.path.join(PLOTS_DIR, '1_target_distribution_original.png')}")

# Plot 2: Distribution of Log-Transformed Transport_Cost
y_transformed = np.log1p(df_train['Transport_Cost'].clip(lower=0))
plt.figure(figsize=(12, 6))
sns.histplot(y_transformed, kde=True, bins=100)
plt.title('Distribution of log1p(Transport_Cost.clip(0))')
plt.xlabel('Log-Transformed Transport_Cost')
plt.ylabel('Frequency')
plt.savefig(os.path.join(PLOTS_DIR, '2_target_distribution_log_transformed.png'))
print(f"Saved plot: {os.path.join(PLOTS_DIR, '2_target_distribution_log_transformed.png')}")

# Plot 3: Correlation Heatmap
plt.figure(figsize=(12, 8))
numerical_cols = df_train.select_dtypes(include=np.number).columns
sns.heatmap(df_train[numerical_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Numerical Feature Correlation Heatmap')
plt.savefig(os.path.join(PLOTS_DIR, '3_numerical_correlation_heatmap.png'))
print(f"Saved plot: {os.path.join(PLOTS_DIR, '3_numerical_correlation_heatmap.png')}")

# Plot 4 & 5: Categorical Feature Analysis
categorical_features = ['Transport_Method', 'Hospital_Info', 'Equipment_Type']

for col in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df_train[col], order=df_train[col].value_counts().index)
    plt.title(f'Count of {col}')
    plt.savefig(os.path.join(PLOTS_DIR, f'4_count_{col}.png'))
    print(f"Saved plot: {os.path.join(PLOTS_DIR, f'4_count_{col}.png')}")

    plt.figure(figsize=(12, 7))
    sns.boxplot(y=col, x=y_transformed, data=df_train, order=df_train[col].value_counts().index)
    plt.title(f'log1p(Transport_Cost) vs. {col}')
    plt.xlabel('log1p(Transport_Cost)')
    plt.savefig(os.path.join(PLOTS_DIR, f'5_boxplot_{col}_vs_target.png'))
    print(f"Saved plot: {os.path.join(PLOTS_DIR, f'5_boxplot_{col}_vs_target.png')}")

print(f"\nEDA complete. Plots are saved in the '{PLOTS_DIR}' directory.")