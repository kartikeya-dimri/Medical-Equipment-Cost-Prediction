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

def preprocess_data_v3(train_path=os.path.join(DATA_DIR, 'train.csv'), 
                       test_path=os.path.join(DATA_DIR, 'test.csv')):
    """
    Loads and preprocesses data using ONLY Equipment_Weight and Equipment_Value.
    - Clips negative costs to 0.
    - Imputes numerical features with median.
    - Saves as .npy files.
    """
    print("Starting preprocessing v3 (Weight & Value only)...")

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
    # Clips negative costs to 0, then applies log1p transform
    y_train = np.log1p(df_train['Transport_Cost'].clip(lower=0))
    # We drop the target but keep the original df_train to get its length
    
    # --- 2. Combine Dataframes ---
    # We only need the two features of interest
    features_to_keep = ['Equipment_Weight', 'Equipment_Value']
    
    combined_df = pd.concat([
        df_train[features_to_keep], 
        df_test[features_to_keep]
    ], ignore_index=True)

    print(f"Combined data shape (features only): {combined_df.shape}")

    # --- 3. Define Feature Lists for Pipeline ---
    numerical_features = features_to_keep
    categorical_features = [] # No categorical features in this version

    print(f"\nUsing numerical features: {numerical_features}")

    # --- 4. Create Preprocessing Pipelines ---

    # Impute missing numerical data with MEDIAN, then scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # --- 5. Create ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features)
        ],
        remainder='drop' # Drop any other columns (though there shouldn't be any)
    )

    # --- 6. Fit and Transform Data ---
    print("\nFitting preprocessor and transforming data...")
    X_processed_combined = preprocessor.fit_transform(combined_df)
    
    # Get feature names (will just be 'Equipment_Weight' and 'Equipment_Value')
    feature_names = preprocessor.get_feature_names_out()
    print(f"Total features after processing: {len(feature_names)}")
    print(f"Feature names: {feature_names}")

    # --- 7. Split Back into Train and Test ---
    # Use the length of the original df_train to split
    X_train_processed = X_processed_combined[:len(df_train)]
    X_test_processed = X_processed_combined[len(df_train):]

    print(f"\nProcessed X_train shape: {X_train_processed.shape}")
    print(f"Processed y_train shape: {y_train.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}")

    # --- 8. Save Processed Data and Pipeline ---
    # Save as .npy files with a '3' to distinguish them
    np.save(os.path.join(OUTPUT_DIR, 'X_train_processed3.npy'), X_train_processed)
    np.save(os.path.join(OUTPUT_DIR, 'y_train_processed3.npy'), y_train.values)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_processed3.npy'), X_test_processed)

    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor3.joblib'))
    
    print("\nPreprocessing complete!")
    print(f"Saved new files to '{OUTPUT_DIR}' directory:")
    print("- X_train_processed3.npy")
    print("- y_train_processed3.npy")
    print("- X_test_processed3.npy")
    print("- preprocessor3.joblib")

if __name__ == "__main__":
    preprocess_data_v3()