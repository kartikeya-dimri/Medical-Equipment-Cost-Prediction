import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
import os
import joblib  # <-- FIX 1: Added this import

# --- 1. Define Paths Robustly ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(SCRIPT_DIR, '..')

DATA_DIR = os.path.join(PARENT_DIR, 'data')
OUTPUT_DIR = os.path.join(PARENT_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End of Path Fix ---

def preprocess_for_catboost():
    """
    Loads, preprocesses, and saves the datasets specifically for CatBoost.
    - No One-Hot Encoding
    - Imputes categorical with 'Missing'
    - Saves as CSV
    """
    print("Starting CatBoost preprocessing...")

    # Load data
    try:
        df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    except FileNotFoundError:
        print(f"Error: Data files not found in {DATA_DIR}.")
        sys.exit(1)

    # --- 1. Handle Target Variable ---
    y_train = np.log1p(df_train['Transport_Cost'].clip(lower=0))
    df_train['Transport_Cost_Log'] = y_train
    df_train = df_train.drop('Transport_Cost', axis=1)

    # --- 2. Combine Dataframes ---
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    
    combined_df = pd.concat([df_train, df_test], ignore_index=True)

    # --- 3. Feature Engineering ---
    combined_df['Order_Placed_Date'] = pd.to_datetime(combined_df['Order_Placed_Date'], errors='coerce')
    combined_df['Delivery_Date'] = pd.to_datetime(combined_df['Delivery_Date'], errors='coerce')
    combined_df['Delivery_Time_Days'] = (combined_df['Delivery_Date'] - combined_df['Order_Placed_Date']).dt.days

    # --- 4. Define Feature Lists ---
    drop_features = ['Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date']
    
    model_cols = [col for col in combined_df.columns if col not in 
                  ['Hospital_Id', 'Transport_Cost_Log', 'is_train'] + drop_features]
    
    numerical_features = combined_df[model_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = combined_df[model_cols].select_dtypes(exclude=np.number).columns.tolist()

    print(f"\nNumerical features to process: {numerical_features}")
    print(f"Categorical features to process: {categorical_features}")

    # --- 5. Create Preprocessing Pipelines ---
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing'))
    ])

    # --- 6. Create ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # --- 7. Fit and Transform Data ---
    print("\nFitting preprocessor and transforming data...")
    original_cols = combined_df.columns
    processed_data = preprocessor.fit_transform(combined_df)
    processed_cols = preprocessor.get_feature_names_out()
    
    processed_df = pd.DataFrame(processed_data, columns=processed_cols)
    
    passthrough_cols = [col for col in original_cols if col not in numerical_features and col not in categorical_features]
    final_col_order = numerical_features + categorical_features + passthrough_cols
    
    for col in numerical_features:
        processed_df[col] = pd.to_numeric(processed_df[col])

    processed_df = processed_df[final_col_order]

    # --- 8. Split Back into Train and Test ---
    train_final = processed_df[processed_df['is_train'] == 1].copy()
    test_final = processed_df[processed_df['is_train'] == 0].copy()

    # Drop the helper columns
    train_final = train_final.drop(columns=['is_train'])
    test_final = test_final.drop(columns=['is_train', 'Transport_Cost_Log'])
    
    # --- FIX 2: ADD THESE TWO LINES ---
    print(f"Dropping unused passthrough columns: {drop_features}")
    train_final = train_final.drop(columns=drop_features, errors='ignore')
    test_final = test_final.drop(columns=drop_features, errors='ignore')
    # --- END OF FIX ---
    
    # --- 9. Save Processed Data to CSV ---
    train_path = os.path.join(OUTPUT_DIR, 'catboost_train.csv')
    test_path = os.path.join(OUTPUT_DIR, 'catboost_test.csv')

    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)
    
    print("\nPreprocessing complete!")
    print(f"Saved files to '{OUTPUT_DIR}' directory:")
    print(f"- catboost_train.csv (Shape: {train_final.shape})")
    print(f"- catboost_test.csv (Shape: {test_final.shape})")
    
    print("\nCategorical features for CatBoost model:")
    print(categorical_features)
    
    joblib.dump(categorical_features, os.path.join(OUTPUT_DIR, 'catboost_categorical_features.joblib'))
    print("Saved list of categorical features to 'catboost_categorical_features.joblib'")

if __name__ == "__main__":
    preprocess_for_catboost()