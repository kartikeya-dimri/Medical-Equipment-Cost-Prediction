import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
    
    # Load the list of categorical feature names
    categorical_features = joblib.load(os.path.join(OUTPUT_DIR, 'catboost_categorical_features.joblib'))
    
    hospital_ids = df_test['Hospital_Id']

except FileNotFoundError:
    print(f"Error: Could not find files in '{OUTPUT_DIR}'.")
    print("Please run 'preprocessing/catboost_preprocessing.py' first.")
    sys.exit(1)

print(f"Loaded train data shape: {df_train_full.shape}")
print(f"Loaded test data shape: {df_test.shape}")
print(f"Loaded {len(categorical_features)} categorical features: {categorical_features}")

# --- 2. Prepare Data for Model ---
# Separate target (y) from features (X)
y_train_full = df_train_full['Transport_Cost_Log']
X_train_full = df_train_full.drop(columns=['Hospital_Id', 'Transport_Cost_Log'])

# Keep only the feature columns in the test set
X_test = df_test.drop(columns=['Hospital_Id'])

# Ensure column order is identical
X_test = X_test[X_train_full.columns]

# --- 3. Create Train/Validation Split for Early Stopping ---
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)
print(f"New train split shape: {X_train.shape}")
print(f"Validation split shape: {X_val.shape}")

# --- 4. Define Tuned CatBoost Model ---
tuned_model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='RMSE',
    random_seed=42,
    verbose=100,            # Print progress every 100 trees
    early_stopping_rounds=50, # Stop if validation score doesn't improve
    cat_features=categorical_features # <-- This is the key part!
)

# --- 5. Train Model with Early Stopping ---
print("\nTraining tuned CatBoost model with native categorical features...")
tuned_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val) # Provide the validation set
)

print("Training complete.")
print(f"Best iteration (number of trees): {tuned_model.get_best_iteration()}")
print(f"Best validation RMSE: {tuned_model.get_best_score()['validation']['RMSE']:.4f}")

# --- 6. Make Predictions on Test Set ---
print("\nMaking predictions on the test set...")
y_pred_log = tuned_model.predict(X_test)
y_pred_original = np.expm1(y_pred_log) # Inverse of log1p
y_pred_final = y_pred_original.clip(min=0) 
print("Predictions generated.")

# --- 7. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, 'catboost_native_tuned.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")

print("\n--- Submission File Head ---")
print(submission_df.head())