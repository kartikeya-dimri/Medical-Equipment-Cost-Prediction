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

def preprocess_data_v4(train_path=os.path.join(DATA_DIR, 'train.csv'), 
                       test_path=os.path.join(DATA_DIR, 'test.csv')):
    """
    Loads, preprocesses, and saves the datasets with advanced feature engineering.
    - Flips dates, clips costs
    - Engineers Location, Date, and Interaction features
    - Manually maps Ordinal and Binary features
    - Saves as .npy files.
    """
    print("Starting preprocessing v4 (Advanced Feature Engineering)...")

    # Load data
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: {train_path} or {test_path} not found.")
        sys.exit(1)

    # --- 1. Handle Target Variable ---
    y_train = np.log1p(df_train['Transport_Cost'].clip(lower=0))
    df_train = df_train.drop('Transport_Cost', axis=1)

    # --- 2. Combine Dataframes ---
    combined_df = pd.concat([df_train, df_test], ignore_index=True)

    # --- 3. Manual Mapping (Ordinal & Binary) ---
    print("Mapping binary and ordinal features...")
    # Map binary
    binary_cols = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 'Fragile_Equipment', 'Rural_Hospital']
    for col in binary_cols:
        combined_df[col] = combined_df[col].map({'Yes': 1, 'No': 0})
    
    # Map ordinal
    combined_df['Hospital_Info'] = combined_df['Hospital_Info'].map({'Wealthy': 2, 'Working Class': 1})
    # We will let the numerical imputer handle NaNs here (e.g., with median)

    # --- 4. Feature Engineering ---
    print("Engineering new features...")
    
    # --- Date Flipping & Engineering ---
    combined_df['Order_Placed_Date'] = pd.to_datetime(combined_df['Order_Placed_Date'], errors='coerce')
    combined_df['Delivery_Date'] = pd.to_datetime(combined_df['Delivery_Date'], errors='coerce')
    
    inverted_mask = combined_df['Delivery_Date'] < combined_df['Order_Placed_Date']
    print(f"Found {inverted_mask.sum()} inverted dates. Swapping them.")
    order_dates = combined_df.loc[inverted_mask, 'Order_Placed_Date']
    delivery_dates = combined_df.loc[inverted_mask, 'Delivery_Date']
    combined_df.loc[inverted_mask, 'Order_Placed_Date'] = delivery_dates
    combined_df.loc[inverted_mask, 'Delivery_Date'] = order_dates

    combined_df['Delivery_Time_Days'] = (combined_df['Delivery_Date'] - combined_df['Order_Placed_Date']).dt.days
    combined_df['Order_Month'] = combined_df['Order_Placed_Date'].dt.month
    combined_df['Order_Year'] = combined_df['Order_Placed_Date'].dt.year

    # --- Location Engineering ---
    location_str = combined_df['Hospital_Location'].astype(str)
    combined_df['Is_Military_Address'] = location_str.str.contains(r'\b(APO|FPO)\b').astype(int)
    combined_df['State'] = location_str.str.extract(r',\s([A-Z]{2})\s')
    # Fill state for military addresses
    combined_df.loc[combined_df['Is_Military_Address'] == 1, 'State'] = 'MIL'
    
    # --- Interaction Engineering ---
    # Impute weight and value *before* division to avoid errors
    weight_median = combined_df['Equipment_Weight'].median()
    combined_df['Equipment_Weight'] = combined_df['Equipment_Weight'].replace(0, np.nan).fillna(weight_median)
    value_median = combined_df['Equipment_Value'].median()
    combined_df['Equipment_Value'] = combined_df['Equipment_Value'].fillna(value_median)
    
    combined_df['Value_Per_Weight'] = combined_df['Equipment_Value'] / combined_df['Equipment_Weight']
    # Replace infinite values (if any) with NaN to be imputed
    combined_df['Value_Per_Weight'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- 5. Define Feature Lists for Pipeline ---
    drop_features = ['Hospital_Id', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date']
    # We drop Supplier_Name here, but Frequency/Target encoding is a great next step
    drop_features.append('Supplier_Name')
                     
    all_cols = combined_df.drop(columns=drop_features).columns
    
    # 'categorical_features' are text columns that need OHE
    categorical_features = ['Equipment_Type', 'Transport_Method', 'State']
    
    # 'numerical_features' are all other columns (including our new engineered ones)
    numerical_features = [col for col in all_cols if col not in categorical_features]

    print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
    print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")

    # --- 6. Create Preprocessing Pipelines ---
    # Impute with median (good for numerical and our mapped 0/1/2 features)
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Impute with most frequent (good for text categories)
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # --- 7. Create ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop' # Drops the columns in 'drop_features'
    )

    # --- 8. Fit and Transform Data ---
    print("\nFitting preprocessor and transforming data...")
    X_processed_combined = preprocessor.fit_transform(combined_df)
    feature_names = preprocessor.get_feature_names_out()
    print(f"Total features after processing: {len(feature_names)}")

    # --- 9. Split Back into Train and Test ---
    X_train_processed = X_processed_combined[:len(df_train)]
    X_test_processed = X_processed_combined[len(df_train):]

    print(f"\nProcessed X_train shape: {X_train_processed.shape}")
    print(f"Processed y_train shape: {y_train.shape}")
    print(f"Processed X_test shape: {X_test_processed.shape}")

    # --- 10. Save Processed Data and Pipeline ---
    np.save(os.path.join(OUTPUT_DIR, 'X_train_processed4.npy'), X_train_processed)
    np.save(os.path.join(OUTPUT_DIR, 'y_train_processed4.npy'), y_train.values)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_processed4.npy'), X_test_processed)
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, 'preprocessor4.joblib'))
    
    print("\nPreprocessing v4 complete!")
    print(f"Saved new files to '{OUTPUT_DIR}' directory:")
    print("- X_train_processed4.npy")
    print("- y_train_processed4.npy")
    print("- X_test_processed4.npy")
    print("- preprocessor4.joblib")

if __name__ == "__main__":
    preprocess_data_v4()