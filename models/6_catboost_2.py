import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

# You must install catboost first: pip install catboost
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("Error: CatBoost library not found.")
    print("Please install it by running: pip install catboost")
    sys.exit(1)

# --- 1. Define Paths and Load NEW Data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

print("Loading preprocessed data (v2)...")
try:
    # Load the new files from preprocessing2.py
    X_train_full = np.load(os.path.join(OUTPUT_DIR, 'X_train_processed2.npy'))
    y_train_full = np.load(os.path.join(OUTPUT_DIR, 'y_train_processed2.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test_processed2.npy'))
    
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    hospital_ids = df_test['Hospital_Id']

except FileNotFoundError:
    print(f"Error: Could not find '...processed2.npy' files in '{OUTPUT_DIR}'.")
    print("Please run 'preprocessing/preprocessing2.py' first.")
    sys.exit(1)

# --- 2. Create Train/Validation Split for Early Stopping ---
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"Full train shape: {X_train_full.shape}")
print(f"New train split shape: {X_train.shape}")
print(f"Validation split shape: {X_val.shape}")

# --- 3. Define Tuned CatBoost Model ---
# We use the same tuned parameters as the '11_catboost_tuned.py' script
# Note: We do NOT pass 'cat_features' because this data is one-hot encoded
tuned_model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='RMSE',
    random_seed=42,
    verbose=100,         # Print progress every 100 trees
    early_stopping_rounds=50 # Stop if validation score doesn't improve
)

# --- 4. Train Model with Early Stopping ---
print("\nTraining tuned CatBoost model with early stopping (on v2 data)...")
tuned_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val) # Provide the validation set
)

print("Training complete.")
print(f"Best iteration (number of trees): {tuned_model.get_best_iteration()}")
print(f"Best validation RMSE: {tuned_model.get_best_score()['validation']['RMSE']:.4f}")

# --- 5. Make Predictions on Test Set ---
print("\nMaking predictions on the test set...")
y_pred_log = tuned_model.predict(X_test)
y_pred_original = np.expm1(y_pred_log) # Inverse of log1p
y_pred_final = y_pred_original.clip(min=0) 
print("Predictions generated.")

# --- 6. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, 'catboost_tuned_2.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")

print("\n--- Submission File Head ---")
print(submission_df.head())