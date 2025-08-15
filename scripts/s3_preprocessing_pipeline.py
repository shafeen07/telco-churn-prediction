# -*- coding: utf-8 -]

# Create a fixed version of the preprocessing script

"""
Telco Customer Churn - S3-Integrated Data Preprocessing Pipeline
"""

import pandas as pd
import numpy as np
import boto3
import json
import joblib
import io
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class S3DataProcessorFixed:
    def __init__(self, config_path=None):
        """Initialize S3 data processor with configuration"""
        
        # Load configuration
        if config_path is None:
            config_path = Path.cwd() / 'config' / 'aws_config.json'
        
        self.config = self.load_config(config_path)
        self.bucket_name = self.config['aws']['bucket_name']
        self.region = self.config['aws']['region']
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3', region_name=self.region)
            print(f"[SUCCESS] Connected to S3 bucket: {self.bucket_name}")
        except NoCredentialsError:
            print(f"[ERROR] AWS credentials not found")
            raise
        
        # Processing metadata
        self.processing_metadata = {
            'timestamp': datetime.now().isoformat(),
            'processor_version': '1.0.0',
            'bucket': self.bucket_name
        }
    
    def load_config(self, config_path):
        """Load AWS configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"[CONFIG] Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print(f"[ERROR] Configuration file not found: {config_path}")
            raise
    
    def upload_to_s3_fixed(self, data, s3_key):
        """Fixed upload method that handles the encoding issue"""
        try:
            if isinstance(data, pd.DataFrame):
                # Upload DataFrame as CSV
                csv_buffer = io.StringIO()
                data.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=csv_string.encode('utf-8'),
                    ContentType='text/csv'
                )
            else:
                # Upload serialized object (like sklearn models)
                buffer = io.BytesIO()
                joblib.dump(data, buffer)
                buffer.seek(0)
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=buffer.getvalue(),
                    ContentType='application/octet-stream'
                )
            
            print(f"[UPLOAD] Data uploaded to s3://{self.bucket_name}/{s3_key}")
            return f"s3://{self.bucket_name}/{s3_key}"
            
        except Exception as e:
            print(f"[ERROR] Failed to upload to {s3_key}: {e}")
            raise
    
    def load_data_from_s3(self):
        """Load raw data from S3"""
        print(f"[DATA] Loading data from S3...")
        
        raw_data_key = self.config['data']['raw_data_path']
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=raw_data_key)
            data_bytes = response['Body'].read()
            df = pd.read_csv(io.BytesIO(data_bytes))
            
            print(f"[SUCCESS] Dataset loaded from S3: {df.shape}")
            return df
            
        except Exception as e:
            print(f"[ERROR] Failed to load data from S3: {e}")
            raise
    
    def clean_data(self, df):
        """Clean and fix data quality issues"""
        print(f"\n[PREPROCESSING] Data cleaning...")
        
        df_clean = df.copy()
        
        # Fix TotalCharges - convert to numeric
        if df_clean['TotalCharges'].dtype == 'object':
            print(f"[CLEANING] Converting TotalCharges to numeric...")
            df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            
            # Handle NaN values
            nan_count = df_clean['TotalCharges'].isnull().sum()
            if nan_count > 0:
                median_total = df_clean['TotalCharges'].median()
                df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(
                    df_clean['tenure'].apply(lambda x: 0 if x == 0 else median_total)
                )
                print(f"[CLEANING] Filled {nan_count} NaN values")
        
        # Remove customerID
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
            print(f"[CLEANING] Removed customerID column")
        
        # Standardize categorical values
        categorical_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_columns:
            if col != 'Churn':
                df_clean[col] = df_clean[col].str.strip().str.title()
        
        # Convert SeniorCitizen to categorical
        if 'SeniorCitizen' in df_clean.columns:
            df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
            print(f"[CLEANING] Converted SeniorCitizen to categorical")
        
        print(f"[SUCCESS] Data cleaning completed: {df_clean.shape}")
        return df_clean
    
    def feature_engineering(self, df):
        """Create new features from existing data"""
        print(f"\n[FEATURE ENG] Creating new features...")
        
        df_fe = df.copy()
        
        # Average monthly charge per service
        if all(col in df_fe.columns for col in ['TotalCharges', 'tenure']):
            df_fe['AvgChargePerMonth'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)
            print(f"[FEATURE] Created AvgChargePerMonth")
        
        # Tenure categories
        if 'tenure' in df_fe.columns:
            df_fe['TenureGroup'] = pd.cut(df_fe['tenure'], 
                                         bins=[0, 12, 24, 48, 72], 
                                         labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years'],
                                         include_lowest=True)
            print(f"[FEATURE] Created TenureGroup")
        
        # Convert categorical features to string to avoid issues
        df_fe['TenureGroup'] = df_fe['TenureGroup'].astype(str)
        
        print(f"[SUCCESS] Feature engineering completed: {df_fe.shape}")
        return df_fe
    
    def encode_categorical_variables(self, df, target_col='Churn'):
        """Encode categorical variables for machine learning"""
        print(f"\n[ENCODING] Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        # Separate features and target
        X = df_encoded.drop(target_col, axis=1)
        y = df_encoded[target_col]
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        print(f"[ENCODING] Target variable encoded")
        
        # One-hot encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if categorical_cols:
            X_categorical = pd.get_dummies(X[categorical_cols], prefix_sep='_', drop_first=True)
            X_numerical = X[numerical_cols]
            X_encoded = pd.concat([X_numerical, X_categorical], axis=1)
            print(f"[ENCODING] One-hot encoded {len(categorical_cols)} columns -> {len(X_encoded.columns)} features")
        else:
            X_encoded = X
        
        return X_encoded, y_encoded, label_encoder
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        print(f"\n[SPLITTING] Splitting data...")
        
        config = self.config['preprocessing']
        test_size = config['test_size']
        val_size = config['val_size']
        random_state = config['random_state']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"[SPLITTING] Data split completed:")
        print(f"   Training: {X_train.shape[0]:,} samples")
        print(f"   Validation: {X_val.shape[0]:,} samples")
        print(f"   Test: {X_test.shape[0]:,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale numerical features"""
        print(f"\n[SCALING] Scaling numerical features...")
        
        # Simple approach - scale all numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            scaler = StandardScaler()
            
            X_train_scaled = X_train.copy()
            X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            
            X_val_scaled = X_val.copy()
            X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
            
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
            
            print(f"[SCALING] Scaled {len(numerical_cols)} numerical columns")
            return X_train_scaled, X_val_scaled, X_test_scaled, scaler
        else:
            return X_train, X_val, X_test, None
    
    def save_to_s3(self, X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler=None):
        """Save all processed data to S3"""
        print(f"\n[S3 SAVE] Uploading processed data to S3...")
        
        # Upload datasets
        datasets = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': pd.DataFrame(y_train, columns=['Churn']),
            'y_val': pd.DataFrame(y_val, columns=['Churn']),
            'y_test': pd.DataFrame(y_test, columns=['Churn'])
        }
        
        for name, data in datasets.items():
            s3_key = f"data/processed/{name}.csv"
            self.upload_to_s3_fixed(data, s3_key)
        
        # Upload encoders and scalers
        self.upload_to_s3_fixed(label_encoder, 'models/encoders/label_encoder.pkl')
        
        if scaler is not None:
            self.upload_to_s3_fixed(scaler, 'models/encoders/feature_scaler.pkl')
        
        # Upload feature names
        feature_names = pd.DataFrame(X_train.columns, columns=['feature'])
        self.upload_to_s3_fixed(feature_names, 'data/processed/feature_names.csv')
        
        print(f"[SUCCESS] All processed data uploaded to S3")
    
    def process_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("=== FIXED S3-INTEGRATED TELCO CHURN PREPROCESSING PIPELINE ===")
        
        try:
            # Step 1: Load data from S3
            df = self.load_data_from_s3()
            
            # Step 2: Clean data
            df_clean = self.clean_data(df)
            
            # Step 3: Feature engineering
            df_features = self.feature_engineering(df_clean)
            
            # Step 4: Encode categorical variables
            X_encoded, y_encoded, label_encoder = self.encode_categorical_variables(df_features)
            
            # Step 5: Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_encoded, y_encoded)
            
            # Step 6: Scale features
            X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_val, X_test)
            
            # Step 7: Save everything to S3
            self.save_to_s3(X_train_scaled, X_val_scaled, X_test_scaled, 
                           y_train, y_val, y_test, label_encoder, scaler)
            
            print(f"\n[SUCCESS] === PREPROCESSING PIPELINE COMPLETED ===")
            print(f"[SUMMARY] Final dataset shape: {X_train_scaled.shape}")
            print(f"[SUMMARY] Features created: {len(X_train_scaled.columns)}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run preprocessing pipeline"""
    try:
        processor = S3DataProcessorFixed()
        success = processor.process_pipeline()
        return success
    except Exception as e:
        print(f"[FATAL] Preprocessing failed: {e}")
        return False

if __name__ == "__main__":
    main()


