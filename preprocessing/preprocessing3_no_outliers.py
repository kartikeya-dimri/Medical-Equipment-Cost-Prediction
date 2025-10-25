import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import sys
import os

# --- 1. Define Paths Robustly ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End of Path Fix ---

def preprocess_data_v3_no_outliers(train_path=os.path.join(DATA_DIR, 'train.csv'),
                                     test_path=os.path.join(DATA_DIR, 'test.csv')):
    """
    Loads and preprocesses data using ONLY Equipment_Weight and Equipment_Value.
    - Clips negative costs to 0.
    - Drops outliers based on IQR from the training set.
    - Caps outliers in the test set based on training bounds.
    - Imputes numerical features with median.
    - Saves as .npy files.
    """
    print("Starting preprocessing v3 (Weight & Value only, dropping outliers)...")

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: {train_path} or {test_path} not found.")
        sys.exit(1)

    print(f"Original train shape: {df_train.shape}")
    print(f"Original test shape: {df_test.shape}")

    # --- 1. Handle Target Variable ---
    y_train = np.log1p(df_train['Transport_Cost'].clip(lower=0))
    # Keep track of original train index to align y_train after dropping outliers
    original_train_index = df_train.index

    # --- 2. Define Features ---
    features_to_keep = ['Equipment_Weight', 'Equipment_Value']

    # --- 3. Handle Outliers (using Training Data only) ---
    print("\nCalculating IQR bounds using training data...")
    df_train_outlier_check = df_train[features_to_keep].copy()

    outlier_bounds = {}
    for col in features_to_keep:
        Q1 = df_train_outlier_check[col].quantile(0.25)
        Q3 = df_train_outlier_check[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
        print(f"  {col}: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}")

        # --- Drop outliers from Training Data ---
        outlier_mask_train = (df_train_outlier_check[col] < lower_bound) | (df_train_outlier_check[col] > upper_bound)
        df_train = df_train[~outlier_mask_train]
        print(f"  Dropped {outlier_mask_train.sum()} outlier rows from train based on {col}.")

        # --- Cap outliers in Test Data ---
        lower_cap_mask_test = df_test[col] < lower_bound
        upper_cap_mask_test = df_test[col] > upper_bound
        df_test.loc[lower_cap_mask_test, col] = lower_bound
        df_test.loc[upper_cap_mask_test, col] = upper_bound
        print(f"  Capped {lower_cap_mask_test.sum() + upper_cap_mask_test.sum()} outlier values in test based on {col}.")

    # --- Align y_train after dropping rows ---
    y_train = y_train.loc[df_train.index]
    print(f"\nShape of df_train after outlier removal: {df_train.shape}")
    print(f"Shape of y_train after outlier removal: {y_train.shape}")

    # --- 4. Combine Dataframes (only selected features) ---
    combined_df = pd.concat([
        df_train[features_to_keep],
        df_test[features_to_keep]
    ], ignore_index=True)

    print(f"Combined data shape (features only): {combined_df.shape}")

    # --- 5. Define Feature Lists for Pipeline ---
    numerical_features = features_to_keep
    categorical_features = [] # No categorical features in this version

    print(f"\nUsing numerical features: {numerical_features}")

    # --- 6. Create Preprocessing Pipelines ---
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # --- 7. Create ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features)
        ],
        remainder='drop'
    )

    # --- 8. Fit and Transform Data ---
    # Fit ONLY on the potentially reduced training data rows before combining
    print("\nFitting preprocessor on training data...")
    preprocessor.fit(df_train[features_to_keep]) # Fit only on train

    print("Transforming combined data...")
    X_processed_combined = preprocessor.transform(combined_df) # Transform combined

    feature_names = preprocessor.get_feature_names_out()
    print(f"Total features after processing: {len(feature_names)}")

    # --- 9. Split Back into Train and Test ---
    # Use the length of the processed df_train (after outlier removal) to split
    X_train_processed = X_processed_combined[:len(df_train)]
    X_test_processed = X_processed_combined[len(df_train):]

    print(f"\nProcessed X_train shape: {X_train_processed.shape}")
    print(f"Processed y_train shape: {y_train.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}")

    # --- 10. Save Processed Data and Pipeline ---
    output_suffix = '3_no_outliers'
    np.save(os.path.join(OUTPUT_DIR, f'X_train_processed{output_suffix}.npy'), X_train_processed)
    np.save(os.path.join(OUTPUT_DIR, f'y_train_processed{output_suffix}.npy'), y_train.values)
    np.save(os.path.join(OUTPUT_DIR, f'X_test_processed{output_suffix}.npy'), X_test_processed)

    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, f'preprocessor{output_suffix}.joblib'))

    print("\nPreprocessing complete!")
    print(f"Saved new files to '{OUTPUT_DIR}' directory with suffix '{output_suffix}':")
    print(f"- X_train_processed{output_suffix}.npy")
    print(f"- y_train_processed{output_suffix}.npy")
    print(f"- X_test_processed{output_suffix}.npy")
    print(f"- preprocessor{output_suffix}.joblib")

if __name__ == "__main__":
    preprocess_data_v3_no_outliers()