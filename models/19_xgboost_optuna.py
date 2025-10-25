import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

# You must install xgboost: pip install xgboost
try:
    import xgboost as xgb
except ImportError:
    print("Error: XGBoost library not found.")
    print("Please install it by running: pip install xgboost")
    sys.exit(1)

# You must install optuna: pip install optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Hide logs
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
    """
    
    # Define the search space for parameters
    params = {
        # Use a fixed n_estimators for the CV phase
        # The final model will use early stopping instead
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000), # Reduced max for CV speed
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        # 'early_stopping_rounds': 50 # Cannot easily use this inside cross_val_score
    }

    model = xgb.XGBRegressor(**params)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- THIS IS THE CORRECTED SECTION ---
    scores = -cross_val_score(
        model,
        X_train_full,
        y_train_full,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1 # Make sure n_jobs is set here if desired for CV parallelization
        # No fit_params argument here
    )
    # --- END OF FIX ---
    
    return np.mean(scores)

# --- 3. Run the Optuna Study ---
print("\n--- Starting Optuna Hyperparameter Search for XGBoost (v3 data) ---")
study = optuna.create_study(direction='minimize')
# Consider reducing n_trials slightly if CV takes too long without early stopping
study.optimize(objective, n_trials=50) 

print("\n--- Optuna Search Complete ---")
print(f"Best RMSE (RMSLE) score: {study.best_value:.4f}")
print("Best parameters found:")
# Optuna might have found a large n_estimators, we'll override it for the final fit
best_params_from_study = study.best_params
print(best_params_from_study)

# --- 4. Train Final Model with Best Parameters ---
print("\nTraining final model on full dataset with best parameters...")

# Get best params found by Optuna
final_params = best_params_from_study 
final_params['random_state'] = 42
final_params['n_jobs'] = -1
final_params['tree_method'] = 'hist'
# Set a high n_estimators and rely on early stopping for the final model
final_params['n_estimators'] = 5000 
final_params['early_stopping_rounds'] = 50

# Split data one last time for final early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

final_model = xgb.XGBRegressor(**final_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False # Set to True or 100 to see progress
)
print("Final model training complete.")
print(f"Best iteration found by early stopping: {final_model.best_iteration}")


# --- 5. Make Predictions ---
print("Making predictions on the test set...")
# Use best_iteration to predict if available, ensures consistency
# XGBoost automatically uses the best iteration if early stopping was used during fit
y_pred_log = final_model.predict(X_test) 
y_pred_original = np.expm1(y_pred_log)
y_pred_final = y_pred_original.clip(min=0)
print("Predictions generated.")

# --- 6. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, 'xgboost_optuna_3.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")
print(submission_df.head())