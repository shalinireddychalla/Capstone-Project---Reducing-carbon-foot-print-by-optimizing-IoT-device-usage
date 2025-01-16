import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def multiple_models(dataset_path):
    # Load the data
    df = pd.read_csv(dataset_path)
    
    # Drop unnecessary columns
    df.drop(['Unnamed: 0', 'Date'], axis=1, inplace=True)
    
    # Define column names
    df.columns = ['State', 'City', 'Crop Type', 'Season', 'Temperature (°C)',
                  'Rainfall (mm)', 'Supply Volume (tons)', 'Demand Volume (tons)',
                  'Transportation Cost (₹/ton)', 'Fertilizer Usage (kg/hectare)',
                  'Pest Infestation (0-1)', 'Market Competition (0-1)', 'Price (₹/ton)']

    # Extract object columns for encoding
    object_cols = df.select_dtypes(include='object').columns
    
    # Encode categorical columns
    mappings = {}
    for col in object_cols:
        unique_values = df[col].unique()
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        mappings[col] = mapping
        df[col] = df[col].map(mapping)
    
    # Define input and output columns
    ind_col = [col for col in df.columns if col != 'Price (₹/ton)']
    dep_col = 'Price (₹/ton)'
    
    X = df[ind_col]
    y = df[dep_col]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    # Initialize models
    rf_model = RandomForestRegressor(random_state=0)
    xgb_model = XGBRegressor(random_state=0)
    svr_model = SVR()
    
    # Train models and make predictions
    rf_model.fit(X_train, y_train)
    print("model1 trained")
    xgb_model.fit(X_train, y_train)
    print("model2 trained")
    svr_model.fit(X_train, y_train)
    print("model3 trained")
    
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)
    
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    
    svr_train_pred = svr_model.predict(X_train)
    svr_test_pred = svr_model.predict(X_test)
    
    # Calculate metrics
    results = {
        'random_forest': {
            'train_mse': mean_squared_error(y_train, rf_train_pred),
            'test_mse': mean_squared_error(y_test, rf_test_pred),
            'train_r2': r2_score(y_train, rf_train_pred),
            'test_r2': r2_score(y_test, rf_test_pred)
        },
        'xgboost': {
            'train_mse': mean_squared_error(y_train, xgb_train_pred),
            'test_mse': mean_squared_error(y_test, xgb_test_pred),
            'train_r2': r2_score(y_train, xgb_train_pred),
            'test_r2': r2_score(y_test, xgb_test_pred)
        },
        'svr': {
            'train_mse': mean_squared_error(y_train, svr_train_pred),
            'test_mse': mean_squared_error(y_test, svr_test_pred),
            'train_r2': r2_score(y_train, svr_train_pred),
            'test_r2': r2_score(y_test, svr_test_pred)
        }
    }
    
    return results

# Example usage:
# results = multiple_regression_models('path_to_your_dataset.csv')
# print(results)
