import pandas as pd
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# --- 1. Define Paths and Load Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

print("Loading preprocessed data...")
try:
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train_processed.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train_processed.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test_processed.npy'))
    
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    hospital_ids = df_test['Hospital_Id']

except FileNotFoundError:
    print(f"Error: Could not find processed .npy files in '{OUTPUT_DIR}'.")
    print("Please run '2_preprocessing.py' from the 'preprocessing' folder first.")
    sys.exit(1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# --- 2. Define K-Fold and Early Stopping Vars ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results_rmse = {}
MAX_DEGREE = 10  # Set a reasonable upper limit
previous_rmse = float('inf') # Initialize with infinity

print(f"\n--- Running 5-Fold CV for Polynomial Degrees (Early Stopping) ---")

# --- 3. Loop Through Degrees and Run CV with Early Stopping ---
for degree in range(1, MAX_DEGREE + 1):
    print(f"Evaluating Polynomial Degree: {degree}...")
    
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    scores = -cross_val_score(pipeline, X_train, y_train, 
                              cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    current_rmse = np.mean(scores)
    
    # --- Early Stopping Logic ---
    if current_rmse > previous_rmse:
        print(f"  STOPPING: RMSE increased from {previous_rmse:.4f} to {current_rmse:.4f}.")
        print(f"  Best performance was at degree {degree - 1}.")
        break # Exit the loop
    else:
        # Store the successful result and update previous_rmse
        cv_results_rmse[degree] = current_rmse
        previous_rmse = current_rmse
        print(f"  Mean RMSE (RMSLE) for Degree {degree}: {current_rmse:.4f}\n")

# --- 4. Plot RMSE vs. Degree ---
print("Generating and saving RMSE vs. Degree plot...")
plt.figure(figsize=(10, 6))
plt.plot(list(cv_results_rmse.keys()), list(cv_results_rmse.values()), marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean RMSE (RMSLE)")
plt.title("Polynomial Regression: Degree vs. Cross-Validated RMSE")
if cv_results_rmse:
    plt.xticks(list(cv_results_rmse.keys()))
plt.grid(True, linestyle='--')

plot_path = os.path.join(OUTPUT_DIR, 'polynomial_rmse_vs_degree_early_stop.png')
plt.savefig(plot_path)
print(f"Plot saved to: {plot_path}")

# --- 5. Find and Train Best Model ---
# Check if any degrees were successfully evaluated
if not cv_results_rmse:
    print("Error: No polynomial degrees were successfully evaluated.")
    print("This might happen if even degree 1 performs poorly.")
    sys.exit(1)

# Find the degree with the minimum RMSE from the ones we ran
best_degree = min(cv_results_rmse, key=cv_results_rmse.get)
best_rmse = cv_results_rmse[best_degree]

print(f"\n--- Best Model Selection ---")
print(f"Best Polynomial Degree: {best_degree} (RMSE: {best_rmse:.4f})")
print(f"Training final model on all data using degree {best_degree}...")

# Create and train the best model pipeline
best_model_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
    ('linear', LinearRegression())
])

best_model_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 6. Make Predictions on Test Set ---
print("Making predictions on the test set...")
y_pred_log = best_model_pipeline.predict(X_test)
y_pred_original = np.expm1(y_pred_log) # Inverse of log1p
y_pred_final = y_pred_original.clip(min=0) # Use min=0 for numpy arrays
print("Predictions generated.")

# --- 7. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, 'polynomial_regression_early_stop.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")

print("\n--- Submission File Head ---")
print(submission_df.head())