import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import sys
import os

# --- 1. Define Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_with_added_feature(train_path=os.path.join(DATA_DIR, 'train.csv'), 
                                  test_path=os.path.join(DATA_DIR, 'test.csv')):
    
    FEATURE_NAME = 'TransportMethod'
    print(f"Starting preprocessing (Base + {FEATURE_NAME})...")

    # --- 2. Load Data ---
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: {train_path} or {test_path} not found.")
        sys.exit(1)

    # --- 3. Handle Target ---
    y_train = np.log1p(df_train['Transport_Cost'].clip(lower=0))

    # --- 4. Define Features to Keep ---
    numerical_features = ['Equipment_Weight', 'Equipment_Value']
    categorical_features = ['Transport_Method']
    
    features_to_keep = numerical_features + categorical_features
    print(f"Features: {features_to_keep}")
    
    combined_df = pd.concat([df_train[features_to_keep], 
                             df_test[features_to_keep]], ignore_index=True)

    # --- 5. Create Pipelines ---
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- 6. Create ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    # --- 7. Fit, Transform, and Split ---
    print("Fitting preprocessor...")
    X_processed_combined = preprocessor.fit_transform(combined_df)
    
    X_train_processed = X_processed_combined[:len(df_train)]
    X_test_processed = X_processed_combined[len(df_train):]

    # --- 8. Save Files ---
    print(f"Processed X_train shape: {X_train_processed.shape}")
    np.save(os.path.join(OUTPUT_DIR, f'X_train_processed_{FEATURE_NAME}.npy'), X_train_processed)
    np.save(os.path.join(OUTPUT_DIR, f'y_train_processed_{FEATURE_NAME}.npy'), y_train.values)
    np.save(os.path.join(OUTPUT_DIR, f'X_test_processed_{FEATURE_NAME}.npy'), X_test_processed)
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, f'preprocessor_{FEATURE_NAME}.joblib'))
    
    print(f"Preprocessing complete. Files saved with suffix '_{FEATURE_NAME}.npy'")

if __name__ == "__main__":
    preprocess_with_added_feature()