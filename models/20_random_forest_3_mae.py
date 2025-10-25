import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error # Import MAE

# --- 1. Define Paths and Load NEW Data (v3) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

print("Loading preprocessed data (v3)...")
try:
    # Load the new files from preprocessing3.py
    X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train_processed3.npy'))
    y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train_processed3.npy'))
    X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test_processed3.npy'))

    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    hospital_ids = df_test['Hospital_Id']

except FileNotFoundError:
    print(f"Error: Could not find '...processed3.npy' files in '{OUTPUT_DIR}'.")
    print("Please run 'preprocessing/preprocessing3.py' first.")
    sys.exit(1)

# --- 2. Define Model and Hyperparameter Grid ---
# n_jobs=-1 here tells the Random Forest to use all CPU cores
model = RandomForestRegressor(random_state=42, n_jobs=-1)

# Using the hyperparameter grid you found successful previously
param_grid = {
    'n_estimators': [300, 250],           # Number of trees in the forest
    'max_depth': [5, 10, None],         # Maximum depth of the tree
    'min_samples_split': [2, 4, 6],     # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 3, 4]      # Minimum samples required at a leaf node
}

# --- 3. Set up Grid Search with k=5 CV, using MAE ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kf,
    scoring='neg_mean_absolute_error', # <-- CHANGED SCORING METRIC
    n_jobs=-1, # Use all cores for the Grid Search itself
    verbose=2  # Show progress
)

print("\n--- Running GridSearchCV for Random Forest (v3 data, optimizing MAE) ---")
print(f"Searching {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])} combinations...")
print("This may take a long time...")
grid_search.fit(X_train, y_train)

# --- 4. Show Best Results ---
# The score is negative MAE, so multiply by -1
best_mae = -grid_search.best_score_
best_params = grid_search.best_params_

print("\n--- Grid Search Complete ---")
print(f"Best parameters found (optimizing for MAE): {best_params}")
print(f"Best MAE (on log-scale) score: {best_mae:.4f}")

# The 'grid_search' object is now the best model
best_model = grid_search.best_estimator_

# --- 5. Make Predictions on Test Set ---
print("\nMaking predictions on the test set with the tuned model...")
y_pred_log = best_model.predict(X_test)
y_pred_original = np.expm1(y_pred_log)
y_pred_final = y_pred_original.clip(min=0)
print("Predictions generated.")

# --- 6. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

# Update filename to reflect MAE optimization
submission_path = os.path.join(OUTPUT_DIR, 'random_forest_grid_3_mae.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")
print(submission_df.head())