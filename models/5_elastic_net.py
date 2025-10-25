import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# --- 1. DEFINE YOUR BEST DEGREE ---
#
# !!! IMPORTANT !!!
# SET THIS VALUE to the "Best Polynomial Degree" found when
# you ran the '6_polynomial_early_stopping.py' script.
#
BEST_DEGREE_FROM_PREVIOUS_STEP = 1  # <-- CHANGE THIS
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# --- 2. Define Paths and Load Data ---
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

print(f"Using Polynomial Degree: {BEST_DEGREE_FROM_PREVIOUS_STEP}")

# --- 3. Create Pipeline ---
# This pipeline combines polynomial features with ElasticNet regression
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=BEST_DEGREE_FROM_PREVIOUS_STEP, include_bias=False)),
    ('elastic', ElasticNet(max_iter=10000, random_state=42)) # 'elastic' is the step name
])

# --- 4. Define Hyperparameter Grid for Tuning ---
# We will tune both 'alpha' and 'l1_ratio'
param_grid = {
    'elastic__alpha': [0.001, 0.01, 0.1, 1.0],
    'elastic__l1_ratio': [0.1, 0.5, 0.9] # 0 is Ridge, 1 is Lasso
}

# --- 5. Set up Grid Search with k=5 CV ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV will test every combination of alpha and l1_ratio
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kf,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,  # Use all available CPU cores
    verbose=2   # Show progress
)

print("\n--- Running GridSearchCV to find best params for ElasticNet ---")
grid_search.fit(X_train, y_train)

# --- 6. Show Best Results ---
# The best score will be negative, so we multiply by -1
best_rmse = -grid_search.best_score_
best_params = grid_search.best_params_

print("\n--- Grid Search Complete ---")
print(f"Best parameters found: {best_params}")
print(f"Best RMSE (RMSLE) score: {best_rmse:.4f}")

# The 'grid_search' object is now the best model
best_model = grid_search.best_estimator_

# --- 7. Make Predictions on Test Set ---
print("\nMaking predictions on the test set with the tuned model...")
y_pred_log = best_model.predict(X_test)
y_pred_original = np.expm1(y_pred_log) # Inverse of log1p
y_pred_final = y_pred_original.clip(min=0) # Use min=0 for numpy arrays
print("Predictions generated.")

# --- 8. Create Submission File ---
submission_df = pd.DataFrame({
    'Hospital_Id': hospital_ids,
    'Transport_Cost': y_pred_final
})

submission_path = os.path.join(OUTPUT_DIR, 'elastic_net_tuned.csv')
submission_df.to_csv(submission_path, index=False)
print(f"\nSuccessfully created submission file at: {submission_path}")

print("\n--- Submission File Head ---")
print(submission_df.head())