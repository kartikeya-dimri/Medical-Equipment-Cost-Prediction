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

# --- 1. Define Paths and Load Data ---
# Get the directory where this script (4_linear_regression.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

print("Loading preprocessed data...")
try:
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train_processed.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train_processed.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test_processed.npy'))
    
    # Load original test.csv to get Hospital_Id for submission
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    hospital_ids = df_test['Hospital_Id']

except FileNotFoundError:
    print(f"Error: Could not find processed .npy files in '{OUTPUT_DIR}'.")
    print("Please run '2_preprocessing.py' from the 'preprocessing' folder first.")
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

# --- 5. Plot Graph (Actual vs. Predicted) ---
print("Generating and saving CV plot...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train, y=y_pred_cv, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel("Actual Log(Transport_Cost)")
plt.ylabel("Predicted Log(Transport_Cost)")
plt.title(f"Linear Regression: Actual vs. Predicted (k=5 Cross-Validation)\nRMSE (RMSLE): {rmse:.4f}")

plot_path = os.path.join(OUTPUT_DIR, 'linear_regression_cv_plot.png')
plt.savefig(plot_path)
print(f"Plot saved to: {plot_path}")

# --- 6. Train Full Model for Submission ---
print("\nTraining final model on all training data...")
model.fit(X_train, y_train)
print("Training complete.")

# --- 7. Make Predictions on Test Set ---
print("Making predictions on the test set...")
y_pred_log = model.predict(X_test)
y_pred_original = np.expm1(y_pred_log)

# --- THIS IS THE CORRECTED LINE ---
y_pred_final = y_pred_original.clip(min=0)
# --- END OF FIX ---

print("Predictions generated.")

# --- 8. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, 'linear_regression.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")

print("\n--- Submission File Head ---")
print(submission_df.head())