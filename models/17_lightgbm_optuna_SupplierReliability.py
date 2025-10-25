import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

# You must install lightgbm: pip install lightgbm
try:
    import lightgbm as lgb
except ImportError:
    print("Error: LightGBM library not found.")
    print("Please install it by running: pip install lightgbm")
    sys.exit(1)

# You must install optuna: pip install optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING) # To hide logs
except ImportError:
    print("Error: Optuna library not found.")
    print("Please install it by running: pip install optuna")
    sys.exit(1)

# --- 1. Define Paths and Load BEST Data (SupplierReliability) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

FEATURE_SUFFIX = 'SupplierReliability' # The winning feature suffix

print(f"Loading preprocessed data (Base + {FEATURE_SUFFIX})...")
try:
    # Load the files corresponding to the best feature
    X_train_full = np.load(os.path.join(OUTPUT_DIR, f'X_train_processed_{FEATURE_SUFFIX}.npy'))
    y_train_full = np.load(os.path.join(OUTPUT_DIR, f'y_train_processed_{FEATURE_SUFFIX}.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, f'X_test_processed_{FEATURE_SUFFIX}.npy'))
    
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    hospital_ids = df_test['Hospital_Id']

except FileNotFoundError:
    print(f"Error: Could not find '...processed_{FEATURE_SUFFIX}.npy' files in '{OUTPUT_DIR}'.")
    print(f"Please ensure 'preprocessing/preprocessing_add_{FEATURE_SUFFIX}.py' has been run.")
    sys.exit(1)

print("Data loaded and prepared.")
print(f"Train data shape: {X_train_full.shape}")

# --- 2. Define the Optuna Objective Function ---
def objective(trial):
    """
    This function is called by Optuna for each trial.
    """
    
    # Define the search space for parameters (same as before)
    params = {
        'objective': 'regression_l2', 
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 8, 64), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1 
    }

    model = lgb.LGBMRegressor(**params)
    
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
print(f"\n--- Starting Optuna Hyperparameter Search for LightGBM ({FEATURE_SUFFIX} data) ---")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # Increase n_trials (e.g., 100) for a better search if time permits

print("\n--- Optuna Search Complete ---")
print(f"Best RMSE (RMSLE) score: {study.best_value:.4f}")
print("Best parameters found:")
print(study.best_params)

# --- 4. Train Final Model with Best Parameters ---
print("\nTraining final model on full dataset with best parameters...")

# Get best params
best_params = study.best_params
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['objective'] = 'regression_l2'
best_params['metric'] = 'rmse'
best_params['verbose'] = -1

final_model = lgb.LGBMRegressor(**best_params)

# We can use early stopping on the final fit
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(20, verbose=False)] # Increased patience
)
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

submission_path = os.path.join(OUTPUT_DIR, f'lightgbm_optuna_{FEATURE_SUFFIX}.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")
print(submission_df.head())