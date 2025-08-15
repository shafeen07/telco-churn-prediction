# -*- coding: utf-8 -*-

"""
Telco Customer Churn - Flask API for Model Inference (CLEAN FIXED VERSION)
Author: Shafeen Ahmed
Purpose: REST API for churn prediction deployed on EC2
CRITICAL FIX: Feature name matching between training and inference
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import boto3
import io
from datetime import datetime
from pathlib import Path
import logging
import traceback

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChurnPredictionAPI:
    def __init__(self, config_path=None):
        """Initialize the ML API"""
        
        # Load configuration
        if config_path is None:
            config_path = Path.cwd() / 'config' / 'aws_config.json'
        
        self.config = self.load_config(config_path)
        self.bucket_name = self.config['aws']['bucket_name']
        self.region = self.config['aws']['region']
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client('s3', region_name=self.region)
            logger.info(f"Connected to S3 bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"S3 connection failed: {e}")
            raise
        
        # Model components
        self.model = None
        self.label_encoder = None
        self.feature_scaler = None
        self.feature_names = None
        self.model_info = None
        self.training_sample = None
        
        # API metrics
        self.prediction_count = 0
        self.startup_time = datetime.now()
        
        # Load model and preprocessors
        self.load_model_components()
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
    
    def download_from_s3(self, s3_key):
        """Download model components from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            if s3_key.endswith('.csv'):
                data_bytes = response['Body'].read()
                return pd.read_csv(io.BytesIO(data_bytes))
            elif s3_key.endswith('.json'):
                data_bytes = response['Body'].read()
                return json.loads(data_bytes.decode('utf-8'))
            else:
                data_bytes = response['Body'].read()
                return joblib.load(io.BytesIO(data_bytes))
                
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise
    
    def find_latest_model(self):
        """Find the latest trained model in S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='models/trained/micro_best_model_'
            )
            
            if 'Contents' not in response:
                raise Exception("No trained models found in S3")
            
            model_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.pkl')]
            if not model_files:
                raise Exception("No .pkl model files found")
            
            latest_model = sorted(model_files)[-1]
            model_filename = latest_model.split('/')[-1]
            timestamp_part = model_filename.replace('micro_best_model_', '').replace('.pkl', '')
            info_file = f'models/trained/micro_best_model_info_{timestamp_part}.json'
            
            logger.info(f"Latest model found: {latest_model}")
            return latest_model, info_file
            
        except Exception as e:
            logger.error(f"Failed to find latest model: {e}")
            raise
    
    def load_model_components(self):
        """Load model and preprocessing components from S3"""
        try:
            logger.info("Loading model components from S3...")
            
            # Find and load the latest model
            model_path, info_path = self.find_latest_model()
            
            # Load model
            self.model = self.download_from_s3(model_path)
            logger.info(f"Model loaded: {type(self.model).__name__}")
            
            # Load model info
            self.model_info = self.download_from_s3(info_path)
            logger.info(f"Model info loaded: {self.model_info['model_name']}")
            
            # Load encoders
            self.label_encoder = self.download_from_s3('models/encoders/label_encoder.pkl')
            logger.info("Label encoder loaded")
            
            # Try to load feature scaler (optional)
            try:
                self.feature_scaler = self.download_from_s3('models/encoders/feature_scaler.pkl')
                logger.info("Feature scaler loaded")
            except:
                logger.info("No feature scaler found (optional)")
            
            # Load feature names
            feature_df = self.download_from_s3('data/processed/feature_names.csv')
            self.feature_names = feature_df['feature'].tolist()
            logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            
            # CRITICAL: Log the expected features for debugging
            logger.info("=== EXPECTED FEATURE ANALYSIS ===")
            expected_categorical_features = [f for f in self.feature_names if '_' in f]
            expected_prefixes = {}
            for feature in expected_categorical_features:
                prefix = feature.split('_')[0]
                if prefix not in expected_prefixes:
                    expected_prefixes[prefix] = []
                expected_prefixes[prefix].append(feature.split('_', 1)[1])
            
            for prefix, values in expected_prefixes.items():
                logger.info(f"Expected {prefix} features: {values}")
            
            # Log some sample expected features
            logger.info(f"Sample expected features: {self.feature_names[:15]}")
            if len(self.feature_names) > 15:
                logger.info(f"... and {len(self.feature_names) - 15} more")
            
            # Load training sample for feature template
            try:
                logger.info("Attempting to load training sample from S3...")
                training_sample = self.download_from_s3('data/processed/X_train.csv')
                
                if training_sample is not None and not training_sample.empty:
                    self.training_sample = training_sample.head(10)
                    logger.info(f"Training sample loaded successfully: {self.training_sample.shape}")
                    logger.info(f"Sample feature names: {self.feature_names[:10]}...")
                else:
                    logger.warning("Training sample is empty or None")
                    self.training_sample = None
                    
            except Exception as e:
                logger.warning(f"Could not load training sample from S3: {e}")
                self.training_sample = None
            
            logger.info("All model components loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model components: {e}")
            raise

    def extract_training_categories_from_features(self):
        """Extract the exact categorical values used during training from feature names"""
        logger.info("=== REVERSE ENGINEERING TRAINING CATEGORIES ===")
        
        # Extract categorical features (those with underscores)
        categorical_features = [f for f in self.feature_names if '_' in f and not f.startswith(('tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargePerMonth'))]
        
        # Group by prefix to understand categories
        training_categories = {}
        for feature in categorical_features:
            parts = feature.split('_', 1)
            if len(parts) == 2:
                prefix, value = parts
                if prefix not in training_categories:
                    training_categories[prefix] = []
                training_categories[prefix].append(value)
        
        # Since we use drop_first=True, we need to add the dropped category back
        # The dropped category is typically the first alphabetically
        for prefix, values in training_categories.items():
            if prefix in ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                # Binary categories - if we see 'Yes', the dropped one is 'No'
                if 'Yes' in values:
                    values.insert(0, 'No')
                elif 'Male' in values:  # Special case for gender
                    values.insert(0, 'Female')
            else:
                # Multi-category features - add the first alphabetically
                sorted_values = sorted(values)
                if prefix == 'Contract':
                    values.insert(0, 'Month-To-Month')  # Likely the most common
                elif prefix == 'InternetService':
                    values.insert(0, 'Dsl')  # Alphabetically first
                elif prefix == 'PaymentMethod':
                    values.insert(0, 'Bank Transfer (Automatic)')  # Alphabetically first
                elif prefix == 'TenureGroup':
                    values.insert(0, '0-1 Year')  # Alphabetically first
                else:
                    # For service-related features, 'No' is typically first
                    values.insert(0, 'No')
        
        logger.info("Reverse-engineered training categories:")
        for prefix, values in training_categories.items():
            logger.info(f"  {prefix}: {values}")
        
        return training_categories

    def debug_feature_mismatch(self, generated_features):
        """Debug feature mismatches by comparing with expected features"""
        expected_features = set(self.feature_names)
        generated_features = set(generated_features)
        
        missing_features = expected_features - generated_features
        extra_features = generated_features - expected_features
        
        logger.info("=== FEATURE MATCHING DEBUG ===")
        logger.info(f"Expected features: {len(expected_features)}")
        logger.info(f"Generated features: {len(generated_features)}")
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features:")
            for i, feature in enumerate(sorted(missing_features)[:10]):
                logger.warning(f"  {i+1}. {feature}")
            if len(missing_features) > 10:
                logger.warning(f"  ... and {len(missing_features) - 10} more")
        
        if extra_features:
            logger.warning(f"Extra {len(extra_features)} features:")
            for i, feature in enumerate(sorted(extra_features)[:10]):
                logger.warning(f"  {i+1}. {feature}")
            if len(extra_features) > 10:
                logger.warning(f"  ... and {len(extra_features) - 10} more")
        
        # Analyze patterns in missing features
        if missing_features:
            logger.info("=== MISSING FEATURE ANALYSIS ===")
            prefixes = {}
            for feature in missing_features:
                if '_' in feature:
                    prefix = feature.split('_')[0]
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(feature.split('_', 1)[1])
            
            for prefix, values in prefixes.items():
                logger.info(f"Missing {prefix} values: {values[:5]}")
        
        return missing_features, extra_features
        """Debug feature mismatches by comparing with expected features"""
        expected_features = set(self.feature_names)
        generated_features = set(generated_features)
        
        missing_features = expected_features - generated_features
        extra_features = generated_features - expected_features
        
        logger.info("=== FEATURE MATCHING DEBUG ===")
        logger.info(f"Expected features: {len(expected_features)}")
        logger.info(f"Generated features: {len(generated_features)}")
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features:")
            for i, feature in enumerate(sorted(missing_features)[:10]):
                logger.warning(f"  {i+1}. {feature}")
            if len(missing_features) > 10:
                logger.warning(f"  ... and {len(missing_features) - 10} more")
        
        if extra_features:
            logger.warning(f"Extra {len(extra_features)} features:")
            for i, feature in enumerate(sorted(extra_features)[:10]):
                logger.warning(f"  {i+1}. {feature}")
            if len(extra_features) > 10:
                logger.warning(f"  ... and {len(extra_features) - 10} more")
        
        # Analyze patterns in missing features
        if missing_features:
            logger.info("=== MISSING FEATURE ANALYSIS ===")
            prefixes = {}
            for feature in missing_features:
                if '_' in feature:
                    prefix = feature.split('_')[0]
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(feature.split('_', 1)[1])
            
            for prefix, values in prefixes.items():
                logger.info(f"Missing {prefix} values: {values[:5]}")
        
        return missing_features, extra_features

    def preprocess_input(self, input_data):
        """Preprocess input data with EXACT training replication"""
        try:
            # Convert to DataFrame if it's a dict
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data)
            
            logger.info(f"Input data shape: {df.shape}")
            logger.info(f"Input columns: {df.columns.tolist()}")
            
            # EXACT REPLICATION of training preprocessing
            df_clean = df.copy()
            
            # Fix TotalCharges - convert to numeric (exactly like training)
            if 'TotalCharges' in df_clean.columns:
                if df_clean['TotalCharges'].dtype == 'object':
                    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
                    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
                    
                    # Handle NaN values exactly like training
                    if df_clean['TotalCharges'].isnull().sum() > 0:
                        median_total = 1397.475  # Approximate median from training data
                        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(
                            df_clean['tenure'].apply(lambda x: 0 if x == 0 else median_total)
                        )
            
            # Remove customerID if present
            if 'customerID' in df_clean.columns:
                df_clean = df_clean.drop('customerID', axis=1)
            
            # Convert SeniorCitizen to categorical (exactly like training)
            if 'SeniorCitizen' in df_clean.columns:
                df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
                logger.info(f"SeniorCitizen converted: {df_clean['SeniorCitizen'].iloc[0]}")
            
            # CRITICAL FIX: Apply .str.title() to ALL categorical columns EXACTLY like training
            categorical_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_columns:
                if col in df_clean.columns:
                    original_val = df_clean[col].iloc[0] if len(df_clean) > 0 else 'N/A'
                    df_clean[col] = df_clean[col].str.strip().str.title()
                    new_val = df_clean[col].iloc[0] if len(df_clean) > 0 else 'N/A'
                    logger.info(f"Column {col}: '{original_val}' -> '{new_val}'")
            
            # Feature engineering (exactly like training)
            df_fe = df_clean.copy()
            
            # Average monthly charge per service
            if all(col in df_fe.columns for col in ['TotalCharges', 'tenure']):
                df_fe['AvgChargePerMonth'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)
                logger.info(f"AvgChargePerMonth created: {df_fe['AvgChargePerMonth'].iloc[0]}")
            
            # Tenure categories (exactly like training)
            if 'tenure' in df_fe.columns:
                df_fe['TenureGroup'] = pd.cut(df_fe['tenure'], 
                                             bins=[0, 12, 24, 48, 72], 
                                             labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years'],
                                             include_lowest=True)
                df_fe['TenureGroup'] = df_fe['TenureGroup'].astype(str)
                logger.info(f"TenureGroup created: {df_fe['TenureGroup'].iloc[0]}")
            
            # Log categorical values before encoding
            categorical_cols = df_fe.select_dtypes(include=['object']).columns.tolist()
            logger.info("Final categorical values before encoding:")
            for col in categorical_cols:
                if len(df_fe) > 0:
                    logger.info(f"  {col}: '{df_fe[col].iloc[0]}'")
            
            # Separate features (no target variable in inference)
            X = df_fe.copy()
            
            # One-hot encode categorical variables (EXACTLY like training)
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if categorical_cols:
                logger.info(f"Encoding {len(categorical_cols)} categorical columns...")
                
                # CRITICAL FIX: Use reverse-engineered categories from training
                training_categories = self.extract_training_categories_from_features()
                
                # Create a comprehensive template with all known training values
                template_rows = []
                
                # Create rows covering all possible combinations from training
                if training_categories:
                    logger.info("Creating template based on reverse-engineered training categories")
                    
                    # Create template rows that cover all the categorical combinations seen in training
                    base_numerical = {col: 0 for col in numerical_cols}
                    
                    # Add rows for each category to ensure all features are generated
                    for prefix, values in training_categories.items():
                        for value in values:
                            template_row = base_numerical.copy()
                            # Set all categorical columns to default 'No' or first value
                            for cat_col in categorical_cols:
                                if cat_col == prefix:
                                    template_row[cat_col] = value
                                elif cat_col in training_categories:
                                    template_row[cat_col] = training_categories[cat_col][0]  # First (dropped) value
                                else:
                                    template_row[cat_col] = 'No'  # Default fallback
                            template_rows.append(template_row)
                
                # Add actual inference data
                actual_data = X[categorical_cols + numerical_cols].to_dict('records')
                
                # Combine template and actual data
                combined_data = template_rows + actual_data
                X_combined = pd.DataFrame(combined_data)
                
                # Ensure consistent string formatting for categorical columns
                for col in categorical_cols:
                    X_combined[col] = X_combined[col].astype(str)
                
                # Apply one-hot encoding
                X_categorical = pd.get_dummies(X_combined[categorical_cols], prefix_sep='_', drop_first=True)
                X_numerical = X_combined[numerical_cols]
                X_encoded_combined = pd.concat([X_numerical, X_categorical], axis=1)
                
                # Extract only the actual inference data (skip template rows)
                n_template_rows = len(template_rows)
                X_encoded = X_encoded_combined.iloc[n_template_rows:].reset_index(drop=True)
                
                logger.info(f"Used {n_template_rows} template rows to ensure consistent encoding")
                logger.info(f"One-hot encoded {len(categorical_cols)} columns -> {len(X_encoded.columns)} features")
            else:
                X_encoded = X
            
            logger.info(f"After encoding shape: {X_encoded.shape}")
            
            # DEBUG: Analyze feature mismatches
            missing_features, extra_features = self.debug_feature_mismatch(X_encoded.columns)
            
            # Ensure all required features are present and in correct order
            if missing_features:
                logger.info(f"Adding {len(missing_features)} missing features")
                for feature in missing_features:
                    X_encoded[feature] = 0
            
            # Remove extra features
            if extra_features:
                logger.info(f"Removing {len(extra_features)} extra features")
                X_encoded = X_encoded.drop(columns=extra_features)
            
            # Reorder columns to match training exactly
            X_encoded = X_encoded[self.feature_names]
            
            # Apply feature scaling if available
            if self.feature_scaler:
                numerical_cols_final = X_encoded.select_dtypes(include=[np.number]).columns.tolist()
                if numerical_cols_final:
                    X_encoded[numerical_cols_final] = self.feature_scaler.transform(X_encoded[numerical_cols_final])
                    logger.info(f"Applied scaling to {len(numerical_cols_final)} numerical features")
            
            logger.info(f"Final shape: {X_encoded.shape}, Expected: ({len(X_encoded)}, {len(self.feature_names)})")
            
            return X_encoded.values
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def predict(self, input_data):
        """Make prediction on input data"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(processed_data)
            prediction_class = self.model.predict(processed_data)
            
            # Convert back to original labels
            prediction_labels = self.label_encoder.inverse_transform(prediction_class)
            
            # Format results
            results = []
            for i in range(len(processed_data)):
                result = {
                    'prediction': prediction_labels[i],
                    'churn_probability': float(prediction_proba[i][1]),
                    'no_churn_probability': float(prediction_proba[i][0]),
                    'confidence': float(max(prediction_proba[i])),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
            
            self.prediction_count += len(results)
            logger.info(f"Prediction completed for {len(results)} samples")
            
            return results[0] if len(results) == 1 else results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

# Initialize the API
api = ChurnPredictionAPI()
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    """Home page with interactive prediction form"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telco Churn Prediction API</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header { 
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white; 
                padding: 30px; 
                text-align: center;
            }
            .header h1 { margin: 0; font-size: 2.5em; }
            .header p { margin: 10px 0; opacity: 0.9; }
            .stats { 
                background: rgba(255,255,255,0.1); 
                padding: 15px; 
                border-radius: 10px; 
                margin-top: 20px;
                display: inline-block;
            }
            .main-content { 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 30px; 
                padding: 30px;
            }
            .prediction-form { 
                background: #f8f9fa; 
                padding: 25px; 
                border-radius: 10px; 
                border: 2px solid #e9ecef;
            }
            .form-group { 
                margin-bottom: 15px; 
            }
            .form-group label { 
                display: block; 
                margin-bottom: 5px; 
                font-weight: bold; 
                color: #495057;
            }
            .form-group select, .form-group input { 
                width: 100%; 
                padding: 10px; 
                border: 2px solid #dee2e6; 
                border-radius: 5px; 
                font-size: 14px;
                transition: border-color 0.3s;
            }
            .form-group select:focus, .form-group input:focus { 
                outline: none; 
                border-color: #667eea; 
            }
            .predict-btn { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 25px; 
                font-size: 16px; 
                font-weight: bold; 
                cursor: pointer; 
                width: 100%;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .predict-btn:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .predict-btn:disabled { 
                background: #6c757d; 
                cursor: not-allowed; 
                transform: none;
                box-shadow: none;
            }
            .result-section { 
                background: #ffffff; 
                padding: 25px; 
                border-radius: 10px; 
                border: 2px solid #e9ecef;
            }
            .result-card { 
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center; 
                margin-top: 15px;
                display: none;
            }
            .result-card.churn { 
                background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
            }
            .probability-bar { 
                background: rgba(255,255,255,0.3); 
                height: 10px; 
                border-radius: 5px; 
                margin: 10px 0; 
                overflow: hidden;
            }
            .probability-fill { 
                height: 100%; 
                background: white; 
                border-radius: 5px; 
                transition: width 0.5s ease;
            }
            .loading { 
                display: none; 
                text-align: center; 
                padding: 20px;
            }
            .spinner { 
                border: 3px solid #f3f3f3; 
                border-top: 3px solid #667eea; 
                border-radius: 50%; 
                width: 40px; 
                height: 40px; 
                animation: spin 1s linear infinite; 
                margin: 0 auto;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .api-docs { 
                grid-column: 1 / -1; 
                margin-top: 30px; 
                padding-top: 30px; 
                border-top: 2px solid #e9ecef;
            }
            .endpoint { 
                background: #f8f9fa; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px; 
                border-left: 4px solid #667eea;
            }
            .method { 
                color: #667eea; 
                font-weight: bold; 
                background: white; 
                padding: 4px 8px; 
                border-radius: 4px; 
                display: inline-block;
            }
            @media (max-width: 768px) {
                .main-content { 
                    grid-template-columns: 1fr; 
                    gap: 20px; 
                    padding: 20px;
                }
                .header h1 { font-size: 2em; }
                body { padding: 10px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Telco Churn Prediction</h1>
                <p>Advanced Machine Learning API for Customer Retention Analysis</p>
                <div class="stats">
                    <strong>Model:</strong> {{ model_name }} | 
                    <strong>Accuracy:</strong> {{ accuracy }}% | 
                    <strong>Predictions:</strong> {{ prediction_count }} | 
                    <strong>Uptime:</strong> {{ uptime }}
                </div>
            </div>
            
            <div class="main-content">
                <div class="prediction-form">
                    <h2>🔮 Try Live Prediction</h2>
                    <p>Enter customer details below to predict churn probability:</p>
                    
                    <form id="predictionForm">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div class="form-group">
                                <label>Gender</label>
                                <select name="gender" required>
                                    <option value="Female">Female</option>
                                    <option value="Male">Male</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Senior Citizen</label>
                                <select name="SeniorCitizen" required>
                                    <option value="0">No</option>
                                    <option value="1">Yes</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Partner</label>
                                <select name="Partner" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Dependents</label>
                                <select name="Dependents" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Tenure (months)</label>
                                <input type="number" name="tenure" min="0" max="100" value="12" required>
                            </div>
                            <div class="form-group">
                                <label>Phone Service</label>
                                <select name="PhoneService" required>
                                    <option value="Yes">Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Multiple Lines</label>
                                <select name="MultipleLines" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No phone service">No phone service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Internet Service</label>
                                <select name="InternetService" required>
                                    <option value="DSL">DSL</option>
                                    <option value="Fiber optic">Fiber optic</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Contract</label>
                                <select name="Contract" required>
                                    <option value="Month-to-month">Month-to-month</option>
                                    <option value="One year">One year</option>
                                    <option value="Two year">Two year</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Payment Method</label>
                                <select name="PaymentMethod" required>
                                    <option value="Electronic check">Electronic check</option>
                                    <option value="Mailed check">Mailed check</option>
                                    <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                                    <option value="Credit card (automatic)">Credit card (automatic)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Monthly Charges ($)</label>
                                <input type="number" name="MonthlyCharges" step="0.01" min="0" value="50.85" required>
                            </div>
                            <div class="form-group">
                                <label>Total Charges ($)</label>
                                <input type="text" name="TotalCharges" value="610.20" required>
                            </div>
                        </div>
                        
                        <!-- Additional Services -->
                        <h3>Additional Services</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div class="form-group">
                                <label>Online Security</label>
                                <select name="OnlineSecurity" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Online Backup</label>
                                <select name="OnlineBackup" required>
                                    <option value="Yes">Yes</option>
                                    <option value="No">No</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Device Protection</label>
                                <select name="DeviceProtection" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Tech Support</label>
                                <select name="TechSupport" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Streaming TV</label>
                                <select name="StreamingTV" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Streaming Movies</label>
                                <select name="StreamingMovies" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                    <option value="No internet service">No internet service</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Paperless Billing</label>
                                <select name="PaperlessBilling" required>
                                    <option value="Yes">Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                        </div>
                        
                        <button type="submit" class="predict-btn" id="predictBtn">
                            🔮 Predict Customer Churn
                        </button>
                    </form>
                </div>
                
                <div class="result-section">
                    <h2>📊 Prediction Results</h2>
                    <p>Results will appear here after prediction...</p>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Analyzing customer data...</p>
                    </div>
                    
                    <div class="result-card" id="resultCard">
                        <h3 id="predictionText">Customer will CHURN</h3>
                        <div class="probability-bar">
                            <div class="probability-fill" id="probabilityFill"></div>
                        </div>
                        <p id="probabilityText">Churn Probability: 75%</p>
                        <p id="confidenceText">Confidence: High</p>
                        <p id="recommendationText">Recommendation: Immediate retention action required</p>
                    </div>
                </div>
                
                <div class="api-docs">
                    <h2>📋 API Documentation</h2>
                    <div class="endpoint">
                        <h3><span class="method">GET</span> /health</h3>
                        <p>Check API health status and performance metrics</p>
                    </div>
                    <div class="endpoint">
                        <h3><span class="method">GET</span> /model_info</h3>
                        <p>Get detailed model information and training metrics</p>
                    </div>
                    <div class="endpoint">
                        <h3><span class="method">POST</span> /predict</h3>
                        <p>Predict churn for a single customer (JSON input)</p>
                    </div>
                    <div class="endpoint">
                        <h3><span class="method">POST</span> /batch_predict</h3>
                        <p>Predict churn for multiple customers simultaneously</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('resultCard').style.display = 'none';
                document.getElementById('predictBtn').disabled = true;
                document.getElementById('predictBtn').innerHTML = '⏳ Analyzing...';
                
                // Collect form data
                const formData = new FormData(this);
                const data = {};
                for (let [key, value] of formData.entries()) {
                    // Convert numeric fields
                    if (['SeniorCitizen', 'tenure', 'MonthlyCharges'].includes(key)) {
                        data[key] = parseFloat(value);
                    } else {
                        data[key] = value;
                    }
                }
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        displayPrediction(result.prediction);
                    } else {
                        throw new Error(result.error || 'Prediction failed');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    // Hide loading
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('predictBtn').disabled = false;
                    document.getElementById('predictBtn').innerHTML = '🔮 Predict Customer Churn';
                }
            });
            
            function displayPrediction(prediction) {
                const resultCard = document.getElementById('resultCard');
                const predictionText = document.getElementById('predictionText');
                const probabilityFill = document.getElementById('probabilityFill');
                const probabilityText = document.getElementById('probabilityText');
                const confidenceText = document.getElementById('confidenceText');
                const recommendationText = document.getElementById('recommendationText');
                
                const churnProb = Math.round(prediction.churn_probability * 100);
                const willChurn = prediction.prediction === 'Yes';
                
                // Update display
                if (willChurn) {
                    resultCard.className = 'result-card churn';
                    predictionText.innerHTML = '⚠️ Customer LIKELY TO CHURN';
                    recommendationText.innerHTML = '💡 Recommendation: Immediate retention action required';
                } else {
                    resultCard.className = 'result-card';
                    predictionText.innerHTML = '✅ Customer LIKELY TO STAY';
                    recommendationText.innerHTML = '💡 Recommendation: Continue current service level';
                }
                
                probabilityFill.style.width = churnProb + '%';
                probabilityText.innerHTML = `Churn Probability: ${churnProb}%`;
                
                const confidence = Math.round(prediction.confidence * 100);
                confidenceText.innerHTML = `Confidence: ${confidence}%`;
                
                resultCard.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    
    uptime = datetime.now() - api.startup_time
    return render_template_string(html_template,
                                model_name=api.model_info['model_name'],
                                accuracy=round(api.model_info['performance']['accuracy'] * 100, 1),
                                uptime=str(uptime).split('.')[0],
                                prediction_count=api.prediction_count,
                                startup_time=api.startup_time.strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': api.model is not None,
            'predictions_served': api.prediction_count,
            'uptime_seconds': (datetime.now() - api.startup_time).total_seconds()
        }
        return jsonify(health_status), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        return jsonify(api.model_info), 200
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict churn for a single customer"""
    try:
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        result = api.predict(input_data)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'model_info': {
                'model_name': api.model_info['model_name'],
                'timestamp': api.model_info['timestamp']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def predict_batch():
    """Predict churn for multiple customers"""
    try:
        input_data = request.get_json()
        
        if not input_data or 'customers' not in input_data:
            return jsonify({'error': 'No customer data provided. Use {"customers": [...]} format'}), 400
        
        customers = input_data['customers']
        if not isinstance(customers, list):
            return jsonify({'error': 'Customers must be a list'}), 400
        
        results = api.predict(customers)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results),
            'model_info': {
                'model_name': api.model_info['model_name'],
                'timestamp': api.model_info['timestamp']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Batch prediction failed: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Telco Churn Prediction API...")
        logger.info(f"Model: {api.model_info['model_name']}")
        logger.info(f"Performance: {api.model_info['performance']}")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise