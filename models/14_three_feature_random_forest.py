import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- 1. Define Paths and Feature Suffixes ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

# List of the suffixes corresponding to your preprocessing files
feature_suffixes = [
    'BaseTransportFee',
    'SupplierReliability',
    'DeliveryTime',
    'TransportMethod',
    'UrgentShipping',
    'EquipmentType'
]

# --- 2. Initialize Tracking Variables ---
best_overall_rmse = float('inf')
best_feature_suffix = None
best_params_for_best_suffix = None
results = {} # To store results for each suffix

# --- 3. Define Model and Hyperparameter Grid ---
# Using the grid you found worked well
param_grid = {
    'n_estimators': [300, 400],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3, 4]
}

# K-Fold Cross-Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# --- 4. Loop Through Each Feature Dataset ---
print("--- Starting Search for Best Single Feature to Add ---")
for suffix in feature_suffixes:
    print(f"\n--- Testing Feature: {suffix} ---")
    
    # --- Load Data for this Suffix ---
    try:
        X_train_path = os.path.join(OUTPUT_DIR, f'X_train_processed_{suffix}.npy')
        y_train_path = os.path.join(OUTPUT_DIR, f'y_train_processed_{suffix}.npy')
        
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        print(f"Loaded data for {suffix}. Train shape: {X_train.shape}")

    except FileNotFoundError:
        print(f"Error: Could not find files for suffix '{suffix}'. Skipping.")
        continue

    # --- Set up Grid Search ---
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1 # Less verbose during the loop
    )

    print(f"Running GridSearchCV for {suffix}...")
    grid_search.fit(X_train, y_train)

    # --- Store Results ---
    current_best_rmse = -grid_search.best_score_
    current_best_params = grid_search.best_params_
    results[suffix] = {'rmse': current_best_rmse, 'params': current_best_params}
    
    print(f"Best RMSE for {suffix}: {current_best_rmse:.4f}")
    print(f"Best parameters for {suffix}: {current_best_params}")

    # --- Update Overall Best ---
    if current_best_rmse < best_overall_rmse:
        best_overall_rmse = current_best_rmse
        best_feature_suffix = suffix
        best_params_for_best_suffix = current_best_params
        print(f"*** New best feature found: {suffix} (RMSE: {best_overall_rmse:.4f}) ***")

# --- 5. Train Final Model on Best Dataset ---
print("\n--- Search Complete ---")
if best_feature_suffix is None:
    print("Error: No datasets were successfully processed.")
    sys.exit(1)

print(f"The best feature to add was: {best_feature_suffix}")
print(f"Best overall RMSE achieved: {best_overall_rmse:.4f}")
print(f"Best parameters for this feature set: {best_params_for_best_suffix}")

print(f"\nTraining final model using {best_feature_suffix} data and its best parameters...")

# --- Load the Best Data ---
try:
    X_train_best = np.load(os.path.join(OUTPUT_DIR, f'X_train_processed_{best_feature_suffix}.npy'))
    y_train_best = np.load(os.path.join(OUTPUT_DIR, f'y_train_processed_{best_feature_suffix}.npy'))
    X_test_best = np.load(os.path.join(OUTPUT_DIR, f'X_test_processed_{best_feature_suffix}.npy'))
    
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    hospital_ids = df_test['Hospital_Id']
except FileNotFoundError:
     print(f"Error: Could not reload files for the best suffix '{best_feature_suffix}'.")
     sys.exit(1)

# --- Define and Train the Final Model ---
final_model = RandomForestRegressor(
    **best_params_for_best_suffix, # Use the best params found for this specific dataset
    random_state=42, 
    n_jobs=-1
)

final_model.fit(X_train_best, y_train_best)
print("Final model training complete.")

# --- 6. Make Predictions ---
print("Making predictions on the test set...")
y_pred_log = final_model.predict(X_test_best)
y_pred_original = np.expm1(y_pred_log)
y_pred_final = y_pred_original.clip(min=0)
print("Predictions generated.")

# --- 7. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, f'random_forest_best_single_feature_{best_feature_suffix}.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file: {submission_path}")
print(submission_df.head())