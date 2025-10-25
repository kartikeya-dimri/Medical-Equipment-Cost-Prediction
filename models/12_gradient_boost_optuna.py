import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# You must install optuna: pip install optuna
try:
    import optuna
except ImportError:
    print("Error: Optuna library not found.")
    print("Please install it by running: pip install optuna")
    sys.exit(1)

# --- 1. Define Paths and Load NEW Data (v3) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

print("Loading preprocessed data (v3)...")
try:
    # Load the new files from preprocessing3.py
    X_train_full = np.load(os.path.join(OUTPUT_DIR, 'X_train_processed3.npy'))
    y_train_full = np.load(os.path.join(OUTPUT_DIR, 'y_train_processed3.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test_processed3.npy'))
    
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    hospital_ids = df_test['Hospital_Id']

except FileNotFoundError:
    print(f"Error: Could not find '...processed3.npy' files in '{OUTPUT_DIR}'.")
    print("Please run 'preprocessing/preprocessing3.py' first.")
    sys.exit(1)

print("Data loaded and prepared.")

# --- 2. Define the Optuna Objective Function ---
def objective(trial):
    """
    This function is called by Optuna for each trial.
    It suggests parameters, runs a 5-fold CV, and returns the mean RMSE.
    """
    
    # Define the search space for parameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 42
    }

    model = GradientBoostingRegressor(**params)
    
    # Use k=5 for a robust score
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = -cross_val_score(
        model,
        X_train_full,
        y_train_full,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    return np.mean(scores)

# --- 3. Run the Optuna Study ---
print("\n--- Starting Optuna Hyperparameter Search (on v3 data) ---")
# We want to 'minimize' the RMSE
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # You can increase n_trials for a longer search

print("\n--- Optuna Search Complete ---")
print(f"Best RMSE (RMSLE) score: {study.best_value:.4f}")
print("Best parameters found:")
print(study.best_params)

# --- 4. Train Final Model with Best Parameters ---
print("\nTraining final model on full dataset with best parameters...")

# Get best params and add some defaults back
best_params = study.best_params
best_params['random_state'] = 42

# We don't need early stopping here as Optuna found the best n_estimators
final_model = GradientBoostingRegressor(**best_params)

final_model.fit(X_train_full, y_train_full)
print("Final model training complete.")

# --- 5. Make Predictions ---
print("Making predictions on the test set...")
y_pred_log = final_model.predict(X_test)
y_pred_original = np.expm1(y_pred_log)
y_pred_final = y_pred_original.clip(min=0)
print("Predictions generated.")

# --- 6. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, 'gradient_boosting_optuna_3.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")
print(submission_df.head())