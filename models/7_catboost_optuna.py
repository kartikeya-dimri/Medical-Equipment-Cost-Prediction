import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import optuna

# You must install catboost first: pip install catboost
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("Error: CatBoost library not found.")
    print("Please install it by running: pip install catboost")
    sys.exit(1)

# --- 1. Define Paths and Load Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

print("Loading preprocessed CatBoost data...")
try:
    df_train_full = pd.read_csv(os.path.join(OUTPUT_DIR, 'catboost_train.csv'))
    df_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'catboost_test.csv'))
    categorical_features = joblib.load(os.path.join(OUTPUT_DIR, 'catboost_categorical_features.joblib'))
    hospital_ids = df_test['Hospital_Id']
except FileNotFoundError:
    print(f"Error: Could not find files in '{OUTPUT_DIR}'.")
    print("Please run 'preprocessing/catboost_preprocessing.py' first.")
    sys.exit(1)

# --- 2. Prepare Data for Model ---
y_train_full = df_train_full['Transport_Cost_Log']
X_train_full = df_train_full.drop(columns=['Hospital_Id', 'Transport_Cost_Log'])
X_test = df_test.drop(columns=['Hospital_Id'])
X_test = X_test[X_train_full.columns]

print("Data loaded and prepared.")

# --- 3. Define the Optuna Objective Function ---
def objective(trial):
    """
    This function is called by Optuna for each trial.
    It suggests parameters, runs a 5-fold CV, and returns the mean RMSE.
    """
    
    # Define the search space for parameters
    params = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'od_type': 'Iter',
        'od_wait': 20, # Early stopping rounds
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False,
        'cat_features': categorical_features  # <-- THE FIX IS HERE
    }

    model = CatBoostRegressor(**params)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Run cross-validation
    scores = -cross_val_score(
        model,
        X_train_full,
        y_train_full,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
        # <-- 'fit_params' argument is removed from here
    )
    
    # Return the mean RMSE for this trial
    return np.mean(scores)

# --- 4. Run the Optuna Study ---
print("\n--- Starting Optuna Hyperparameter Search ---")
# We want to 'minimize' the RMSE
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # You can increase n_trials for a longer search

print("\n--- Optuna Search Complete ---")
print(f"Best RMSE (RMSLE) score: {study.best_value:.4f}")
print("Best parameters found:")
print(study.best_params)

# --- 5. Train Final Model with Best Parameters ---
print("\nTraining final model on full dataset with best parameters...")

# Get best params and add some defaults back
best_params = study.best_params
best_params['loss_function'] = 'RMSE'
best_params['random_seed'] = 42
best_params['verbose'] = 100 # Show progress
best_params['iterations'] = 5000 # Use a high number with early stopping
best_params['early_stopping_rounds'] = 50
# 'cat_features' is already in best_params from the study, but we can be explicit
best_params['cat_features'] = categorical_features 

# Split data one last time for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

final_model = CatBoostRegressor(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val)
    # No need to pass cat_features here since it's in best_params
)
print("Final model training complete.")

# --- 6. Make Predictions ---
print("Making predictions on the test set...")
y_pred_log = final_model.predict(X_test)
y_pred_original = np.expm1(y_pred_log)
y_pred_final = y_pred_original.clip(min=0)
print("Predictions generated.")

# --- 7. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, 'catboost_optuna_tuned.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")
print(submission_df.head())