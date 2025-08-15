# -*- coding: utf-8 -*-

"""
Telco Customer Churn - T2.Micro Optimized Training Pipeline
Author: Shafeen Ahmed
Purpose: Memory-efficient training for small EC2 instances
"""

import pandas as pd
import numpy as np
import boto3
import json
import joblib
import io
import gc  # Garbage collection for memory management
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries - Import only what we need
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Model evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, GridSearchCV

# Lightweight visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

class MicroInstanceTrainer:
    def __init__(self, config_path=None):
        """Initialize memory-efficient trainer for t2.micro"""
        
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
        except Exception as e:
            print(f"[ERROR] S3 connection failed: {e}")
            raise
        
        # Training metadata
        self.training_metadata = {
            'timestamp': datetime.now().isoformat(),
            'instance_type': 't2.micro',
            'trainer_version': '1.0.0-micro',
            'bucket': self.bucket_name
        }
        
        # Results storage
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
        print(f"[INFO] Optimized for t2.micro instance")
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"[ERROR] Configuration file not found: {config_path}")
            raise
    
    def download_from_s3(self, s3_key):
        """Download data from S3 with memory optimization"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            if s3_key.endswith('.csv'):
                # Load CSV data with memory optimization
                data_bytes = response['Body'].read()
                df = pd.read_csv(io.BytesIO(data_bytes))
                
                # Optimize data types to save memory
                for col in df.select_dtypes(include=['int64']):
                    df[col] = df[col].astype('int32')
                for col in df.select_dtypes(include=['float64']):
                    df[col] = df[col].astype('float32')
                
                return df
            else:
                # Load pickled objects
                data_bytes = response['Body'].read()
                return joblib.load(io.BytesIO(data_bytes))
                
        except Exception as e:
            print(f"[ERROR] Failed to download {s3_key}: {e}")
            raise
    
    def upload_to_s3(self, data, s3_key):
        """Upload data to S3"""
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
            elif isinstance(data, plt.Figure):
                # Upload matplotlib figure
                img_buffer = io.BytesIO()
                data.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')  # Lower DPI to save memory
                img_buffer.seek(0)
                
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=img_buffer.getvalue(),
                    ContentType='image/png'
                )
                plt.close(data)  # Close figure to free memory
            elif s3_key.endswith('.json'):
                # Upload JSON
                json_string = json.dumps(data, indent=2, default=str)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=json_string.encode('utf-8'),
                    ContentType='application/json'
                )
            else:
                # Upload pickled objects
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
            
        except Exception as e:
            print(f"[ERROR] Failed to upload to {s3_key}: {e}")
            raise
    
    def load_processed_data(self):
        """Load processed data from S3 with memory management"""
        print(f"[DATA] Loading processed data from S3...")
        
        try:
            # Load training data
            print(f"[MEMORY] Loading training data...")
            X_train = self.download_from_s3('data/processed/X_train.csv')
            y_train = self.download_from_s3('data/processed/y_train.csv')['Churn'].values
            
            print(f"[MEMORY] Loading validation data...")
            X_val = self.download_from_s3('data/processed/X_val.csv')
            y_val = self.download_from_s3('data/processed/y_val.csv')['Churn'].values
            
            print(f"[MEMORY] Loading test data...")
            X_test = self.download_from_s3('data/processed/X_test.csv')
            y_test = self.download_from_s3('data/processed/y_test.csv')['Churn'].values
            
            # Load encoders
            label_encoder = self.download_from_s3('models/encoders/label_encoder.pkl')
            
            print(f"[SUCCESS] Data loaded successfully")
            print(f"   Training: {X_train.shape}")
            print(f"   Validation: {X_val.shape}")
            print(f"   Test: {X_test.shape}")
            print(f"   Memory usage: ~{X_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Store data
            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
            self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
            self.label_encoder = label_encoder
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load processed data: {e}")
            return False
    
    def define_micro_models(self):
        """Define lightweight models optimized for t2.micro"""
        print(f"[MODELS] Defining t2.micro optimized models...")
        
        models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=500, solver='liblinear'),
                'params': {
                    'C': [0.1, 1, 10],  # Reduced parameter space
                    'penalty': ['l1', 'l2']
                }
            },
            'Naive Bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-7, 1e-5]  # Simplified grid
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [5, 10, 15],  # Limited depth to control memory
                    'min_samples_split': [5, 10],
                    'criterion': ['gini', 'entropy']
                }
            },
            'Random Forest (Small)': {
                'model': RandomForestClassifier(
                    random_state=42, 
                    n_estimators=50,  # Reduced number of trees
                    max_depth=10,     # Limited depth
                    n_jobs=1          # Single thread to avoid memory issues
                ),
                'params': {
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 5]
                }
            }
        }
        
        print(f"[SUCCESS] Defined {len(models)} lightweight models")
        return models
    
    def evaluate_model(self, model, X_val, y_val, model_name):
        """Memory-efficient model evaluation"""
        
        # Predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_proba) if y_proba is not None else None
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        # Free memory
        del y_pred, y_proba
        gc.collect()
        
        return metrics, cm
    
    def train_single_model(self, model_name, model_config):
        """Train a single model with memory management"""
        print(f"\n[TRAINING] {model_name}...")
        
        model = model_config['model']
        param_grid = model_config['params']
        
        try:
            # Use smaller CV and limited parameter search for t2.micro
            search = GridSearchCV(
                model, param_grid, 
                cv=3,  # Reduced CV folds
                scoring='roc_auc', 
                n_jobs=1,  # Single thread
                verbose=0
            )
            
            # Fit the model
            search.fit(self.X_train, self.y_train)
            
            # Get best model
            best_model = search.best_estimator_
            
            # Quick CV score (3-fold)
            cv_scores = cross_val_score(best_model, self.X_train, self.y_train, 
                                      cv=3, scoring='roc_auc')
            
            # Evaluate on validation set
            metrics, cm = self.evaluate_model(
                best_model, self.X_val, self.y_val, model_name
            )
            
            # Store results
            self.model_results[model_name] = {
                'best_params': search.best_params_,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'validation_metrics': metrics,
                'confusion_matrix': cm.tolist()
            }
            
            # Keep only the best model to save memory
            if (not hasattr(self, 'best_roc_auc') or 
                metrics['roc_auc'] > self.best_roc_auc):
                self.best_roc_auc = metrics['roc_auc']
                self.best_model = best_model
                self.best_model_name = model_name
            
            print(f"[SUCCESS] {model_name} - ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Force garbage collection
            del search
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to train {model_name}: {e}")
            print(f"[INFO] This might be due to memory constraints on t2.micro")
            return False
    
    def train_all_models(self):
        """Train all models sequentially with memory management"""
        print(f"\n[TRAINING] Starting lightweight training pipeline...")
        
        models = self.define_micro_models()
        successful_models = 0
        self.best_roc_auc = 0
        
        for model_name, model_config in models.items():
            print(f"\n[MEMORY] Available before {model_name}: {self.get_memory_usage()}")
            
            success = self.train_single_model(model_name, model_config)
            if success:
                successful_models += 1
            
            # Force garbage collection after each model
            gc.collect()
        
        print(f"\n[SUMMARY] Successfully trained {successful_models}/{len(models)} models")
        
        if self.best_model:
            print(f"[BEST MODEL] {self.best_model_name} - ROC AUC: {self.best_roc_auc:.4f}")
    
    def get_memory_usage(self):
        """Get current memory usage info"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f} MB"
        except ImportError:
            return "psutil not available"
    
    def create_simple_comparison_plot(self):
        """Create memory-efficient comparison plot"""
        if not self.model_results:
            return None
        
        # Prepare data for plotting
        models = list(self.model_results.keys())
        roc_aucs = [self.model_results[model]['validation_metrics']['roc_auc'] 
                   for model in models]
        
        # Create simple bar plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        bars = ax.bar(range(len(models)), roc_aucs, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        
        # Add value labels
        for bar, value in zip(bars, roc_aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title('Model Performance Comparison (ROC AUC)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('ROC AUC Score')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def evaluate_best_model_on_test(self):
        """Final evaluation of best model on test set"""
        if not self.best_model:
            print("[ERROR] No best model found")
            return None
        
        print(f"\n[FINAL TEST] Evaluating {self.best_model_name} on test set...")
        
        # Test set evaluation
        metrics, cm = self.evaluate_model(
            self.best_model, self.X_test, self.y_test, self.best_model_name
        )
        
        self.final_test_results = {
            'model_name': self.best_model_name,
            'test_metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'instance_type': 't2.micro'
        }
        
        print(f"[FINAL RESULTS] {self.best_model_name}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
        
        return self.final_test_results
    
    def save_results_to_s3(self):
        """Save results to S3"""
        print(f"\n[S3 SAVE] Uploading results to S3...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save model comparison results
            self.upload_to_s3(
                self.model_results,
                f'models/experiments/micro_model_comparison_{timestamp}.json'
            )
            
            # Save final test results
            if hasattr(self, 'final_test_results'):
                self.upload_to_s3(
                    self.final_test_results,
                    f'models/experiments/micro_final_test_results_{timestamp}.json'
                )
            
            # Save best model
            if self.best_model:
                self.upload_to_s3(
                    self.best_model,
                    f'models/trained/micro_best_model_{timestamp}.pkl'
                )
                
                # Save model info
                model_info = {
                    'model_name': self.best_model_name,
                    'timestamp': timestamp,
                    'instance_type': 't2.micro',
                    'performance': self.model_results[self.best_model_name]['validation_metrics'],
                    'parameters': self.model_results[self.best_model_name]['best_params']
                }
                self.upload_to_s3(
                    model_info,
                    f'models/trained/micro_best_model_info_{timestamp}.json'
                )
            
            # Save visualization
            if hasattr(self, 'comparison_plot') and self.comparison_plot:
                self.upload_to_s3(
                    self.comparison_plot,
                    f'outputs/plots/micro_model_comparison_{timestamp}.png'
                )
            
            print(f"[SUCCESS] All results uploaded to S3")
            
        except Exception as e:
            print(f"[ERROR] Failed to save results: {e}")
    
    def run_micro_training_pipeline(self):
        """Run the complete training pipeline optimized for t2.micro"""
        print("=== T2.MICRO OPTIMIZED TRAINING PIPELINE ===")
        
        try:
            # Step 1: Load processed data
            if not self.load_processed_data():
                return False
            
            # Step 2: Train models
            self.train_all_models()
            
            # Step 3: Create simple visualization
            print(f"\n[VISUALIZATION] Creating comparison plot...")
            self.comparison_plot = self.create_simple_comparison_plot()
            
            # Step 4: Final evaluation on test set
            self.evaluate_best_model_on_test()
            
            # Step 5: Save results to S3
            self.save_results_to_s3()
            
            print(f"\n[SUCCESS] === T2.MICRO TRAINING PIPELINE COMPLETED ===")
            print(f"[SUMMARY] Best Model: {self.best_model_name}")
            print(f"[SUMMARY] Final ROC AUC: {self.best_roc_auc:.4f}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run micro training pipeline"""
    try:
        trainer = MicroInstanceTrainer()
        success = trainer.run_micro_training_pipeline()
        return success
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        return False

if __name__ == "__main__":
    main()