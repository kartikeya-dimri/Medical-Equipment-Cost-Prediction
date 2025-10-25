import pandas as pd
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Define Paths and Load NEW Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

print("Loading preprocessed data (v2)...")
try:
    # Load the new files from preprocessing2.py
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train_processed2.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train_processed2.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test_processed2.npy'))
    
    # Load original test.csv to get Hospital_Id for submission
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    hospital_ids = df_test['Hospital_Id']

except FileNotFoundError:
    print(f"Error: Could not find '...processed2.npy' files in '{OUTPUT_DIR}'.")
    print("Please run 'preprocessing/preprocessing2.py' first.")
    sys.exit(1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")

# --- 2. Define Model and K-Fold (k=5) ---
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- 3. Run Cross-Validation and Get Predictions ---
print("\nRunning 5-fold cross-validation to get predictions...")
y_pred_cv = cross_val_predict(model, X_train, y_train, cv=kf, n_jobs=-1)
print("Cross-validation complete.")

# --- 4. Calculate and Print RMSE ---
rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
print(f"\nCross-Validated RMSE (on log-scale): {rmse:.4f}")
print("(This score is equivalent to RMSLE on the original data)")

# --- 5. Train Full Model for Submission ---
print("\nTraining final model on all training data...")
model.fit(X_train, y_train)
print("Training complete.")

# --- 6. Make Predictions on Test Set ---
print("Making predictions on the test set...")
y_pred_log = model.predict(X_test)
y_pred_original = np.expm1(y_pred_log)
y_pred_final = y_pred_original.clip(min=0) # Use min=0 for numpy arrays
print("Predictions generated.")

# --- 7. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

# Save to a new file name
submission_path = os.path.join(OUTPUT_DIR, 'linear_regression_2.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")

print("\n--- Submission File Head ---")
print(submission_df.head())