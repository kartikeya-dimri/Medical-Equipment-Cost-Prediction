import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

def preprocess_data_v2(train_path=os.path.join(DATA_DIR, 'train.csv'), 
                       test_path=os.path.join(DATA_DIR, 'test.csv')):
    """
    Loads, preprocesses, and saves the datasets with new logic.
    - Flips inverted dates.
    - Clips negative costs to 0.
    - Imputes numerical features with median.
    - Drops Supplier_Name and Hospital_Location.
    - Saves as .npy files.
    """
    print("Starting preprocessing v2...")

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
    df_train = df_train.drop('Transport_Cost', axis=1)

    # --- 2. Combine Dataframes ---
    combined_df = pd.concat([df_train, df_test], ignore_index=True)

    # --- 3. Feature Engineering & Cleaning ---
    
    # Convert dates
    combined_df['Order_Placed_Date'] = pd.to_datetime(combined_df['Order_Placed_Date'], errors='coerce')
    combined_df['Delivery_Date'] = pd.to_datetime(combined_df['Delivery_Date'], errors='coerce')
    
    # --- NEW: Flip inverted dates ---
    print("Checking for inverted dates...")
    inverted_mask = combined_df['Delivery_Date'] < combined_df['Order_Placed_Date']
    print(f"Found {inverted_mask.sum()} rows with inverted dates. Swapping them.")
    
    # Create temp variables for swapping
    order_dates = combined_df.loc[inverted_mask, 'Order_Placed_Date']
    delivery_dates = combined_df.loc[inverted_mask, 'Delivery_Date']
    
    # Perform the swap
    combined_df.loc[inverted_mask, 'Order_Placed_Date'] = delivery_dates
    combined_df.loc[inverted_mask, 'Delivery_Date'] = order_dates
    # --- End of new logic ---

    # Calculate delivery time
    combined_df['Delivery_Time_Days'] = (combined_df['Delivery_Date'] - combined_df['Order_Placed_Date']).dt.days
    
    # --- 4. Define Feature Lists for Pipeline ---
    
    # NEW: Drop 'Supplier_Name' and 'Hospital_Location' as requested
    drop_features = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 
                     'Order_Placed_Date', 'Delivery_Date']
                     
    all_cols = combined_df.drop(columns=drop_features).columns
    numerical_features = combined_df[all_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_features = combined_df[all_cols].select_dtypes(exclude=np.number).columns.tolist()

    print(f"\nNumerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")

    # --- 5. Create Preprocessing Pipelines ---

    # Impute missing numerical data with MEDIAN
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Impute missing categorical data with MOST FREQUENT
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
        remainder='drop' # Drops the columns in 'drop_features'
    )

    # --- 7. Fit and Transform Data ---
    print("\nFitting preprocessor and transforming data...")
    X_processed_combined = preprocessor.fit_transform(combined_df)
    feature_names = preprocessor.get_feature_names_out()
    print(f"Total features after processing: {len(feature_names)}")

    X_processed_combined_df = pd.DataFrame(X_processed_combined, columns=feature_names)

    # --- 8. Split Back into Train and Test ---
    X_train_processed = X_processed_combined_df.iloc[:len(df_train)]
    X_test_processed = X_processed_combined_df.iloc[len(df_train):]

    print(f"\nProcessed X_train shape: {X_train_processed.shape}")
    print(f"Processed y_train shape: {y_train.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}")

    # --- 9. Save Processed Data and Pipeline ---
    # Save as .npy files with a '2' to distinguish them
    np.save(os.path.join(OUTPUT_DIR, 'X_train_processed2.npy'), X_train_processed.values)
    np.save(os.path.join(OUTPUT_DIR, 'y_train_processed2.npy'), y_train.values)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_processed2.npy'), X_test_processed.values)

    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor2.joblib'))
    
    print("\nPreprocessing complete!")
    print(f"Saved new files to '{OUTPUT_DIR}' directory:")
    print("- X_train_processed2.npy")
    print("- y_train_processed2.npy")
    print("- X_test_processed2.npy")
    print("- preprocessor2.joblib")

if __name__ == "__main__":
    preprocess_data_v2()