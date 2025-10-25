import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import sys
import os

# --- Define Paths Robustly ---
# Get the directory where this script (2_preprocessing.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End of Path Fix ---

def preprocess_data(train_path=os.path.join(DATA_DIR, 'train.csv'), 
                    test_path=os.path.join(DATA_DIR, 'test.csv')):
    """
    Loads, preprocesses, and saves the datasets.
    """
    print("Starting preprocessing...")

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
    df_train = df_train.drop('Transport_Cost', axis=1)

    # --- 2. Feature Engineering ---
    combined_df = pd.concat([df_train, df_test], ignore_index=True)

    combined_df['Order_Placed_Date'] = pd.to_datetime(combined_df['Order_Placed_Date'], errors='coerce')
    combined_df['Delivery_Date'] = pd.to_datetime(combined_df['Delivery_Date'], errors='coerce')
    combined_df['Delivery_Time_Days'] = (combined_df['Delivery_Date'] - combined_df['Order_Placed_Date']).dt.days
    
    # --- 3. Define Feature Lists for Pipeline ---
    drop_features = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 
                     'Order_Placed_Date', 'Delivery_Date']
                     
    all_cols = combined_df.drop(columns=drop_features).columns
    numerical_features = combined_df[all_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = combined_df[all_cols].select_dtypes(exclude=np.number).columns.tolist()

    print(f"\nNumerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # --- 4. Create Preprocessing Pipelines ---
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- 5. Create ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )

    # --- 6. Fit and Transform Data ---
    print("\nFitting preprocessor and transforming data...")
    X_processed_combined = preprocessor.fit_transform(combined_df)
    feature_names = preprocessor.get_feature_names_out()
    print(f"Total features after processing: {len(feature_names)}")

    X_processed_combined_df = pd.DataFrame(X_processed_combined, columns=feature_names)

    # --- 7. Split Back into Train and Test ---
    X_train_processed = X_processed_combined_df.iloc[:len(df_train)]
    X_test_processed = X_processed_combined_df.iloc[len(df_train):]

    print(f"\nProcessed X_train shape: {X_train_processed.shape}")
    print(f"Processed y_train shape: {y_train.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}")

    # --- 8. Save Processed Data and Pipeline ---
    np.save(os.path.join(OUTPUT_DIR, 'X_train_processed.npy'), X_train_processed.values)
    np.save(os.path.join(OUTPUT_DIR, 'y_train_processed.npy'), y_train.values)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_processed.npy'), X_test_processed.values)
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor.joblib'))
    
    print("\nPreprocessing complete!")
    print(f"Saved files to '{OUTPUT_DIR}' directory:")
    print("- X_train_processed.npy")
    print("- y_train_processed.npy")
    print("- X_test_processed.npy")
    print("- preprocessor.joblib")

if __name__ == "__main__":
    preprocess_data()