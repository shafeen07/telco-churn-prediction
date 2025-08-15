"""
Data Exploration with Visualizations - EC2 + S3 Integration
Author: Shafeen Ahmed
Purpose: Run data exploration on EC2 with S3 integration
Reads data from S3, generates visualizations, uploads results back to S3
"""

import pandas as pd
import numpy as np
import boto3
import json
import os
from pathlib import Path
import warnings
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError
import io
import tempfile

warnings.filterwarnings('ignore')

class TelcoChurnS3Explorer:
    def __init__(self, config_path=None):
        """Initialize S3 explorer with configuration"""
        
        print("=== TELCO CUSTOMER CHURN - EC2/S3 DATA EXPLORATION ===")
        
        # Load configuration
        self.config = self.load_config(config_path)
        if not self.config:
            print("[ERROR] Failed to load configuration")
            raise Exception("Configuration required")
        
        self.bucket_name = self.config['aws']['bucket_name']
        self.region = self.config['aws']['region']
        
        print(f"[CONFIG] Bucket: {self.bucket_name}")
        print(f"[CONFIG] Region: {self.region}")
        
        # Initialize AWS clients
        try:
            self.s3_client = boto3.client('s3', region_name=self.region)
            self.s3_resource = boto3.resource('s3', region_name=self.region)
            print(f"[SUCCESS] AWS S3 client initialized")
        except NoCredentialsError:
            print(f"[ERROR] AWS credentials not found")
            raise
        
        # Setup matplotlib for headless environment
        self.setup_matplotlib()
        
        # Create temp directory for local processing
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"[INFO] Temporary directory: {self.temp_dir}")
        
    def load_config(self, config_path=None):
        """Load AWS configuration"""
        
        # Try different config locations
        config_locations = [
            config_path,
            'config/aws_config.json',
            './aws_config.json',
            '../config/aws_config.json'
        ]
        
        for location in config_locations:
            if location and Path(location).exists():
                try:
                    with open(location, 'r') as f:
                        config = json.load(f)
                    print(f"[SUCCESS] Config loaded from: {location}")
                    return config
                except Exception as e:
                    print(f"[WARNING] Failed to load config from {location}: {e}")
                    continue
        
        # Try to download config from S3 if local not found
        try:
            # This requires the bucket name to be known, so we'll try a few common patterns
            potential_buckets = []
            
            # List user's buckets and find telco-churn related ones
            response = self.s3_client.list_buckets()
            for bucket in response['Buckets']:
                if 'telco-churn' in bucket['Name'].lower():
                    potential_buckets.append(bucket['Name'])
            
            for bucket_name in potential_buckets:
                try:
                    response = self.s3_client.get_object(
                        Bucket=bucket_name,
                        Key='config/aws_config.json'
                    )
                    config = json.loads(response['Body'].read().decode('utf-8'))
                    print(f"[SUCCESS] Config downloaded from S3: {bucket_name}")
                    return config
                except ClientError:
                    continue
                    
        except Exception as e:
            print(f"[WARNING] Could not auto-detect S3 config: {e}")
        
        print(f"[ERROR] No valid configuration found")
        return None
    
    def setup_matplotlib(self):
        """Setup matplotlib for EC2/headless environment"""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.plt = plt
        self.sns = sns
        
        print(f"[SUCCESS] Matplotlib configured for headless operation")
    
    def download_data_from_s3(self):
        """Download raw data from S3"""
        print(f"\n[S3] Downloading raw data...")
        
        s3_key = self.config['data']['raw_data_path']
        local_path = self.temp_dir / 'raw_data.csv'
        
        try:
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            
            # Verify download
            if local_path.exists():
                size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"[SUCCESS] Downloaded: {size_mb:.2f} MB")
                return local_path
            else:
                raise Exception("File not found after download")
                
        except ClientError as e:
            print(f"[ERROR] Failed to download data: {e}")
            raise
    
    def upload_plot_to_s3(self, local_path, s3_key, metadata=None):
        """Upload a plot file to S3"""
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Set content type for images
            extra_args['ContentType'] = 'image/png'
            
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            
            print(f"[UPLOADED] s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            print(f"[ERROR] Failed to upload {s3_key}: {e}")
            return False
    
    def create_exploration_report(self, df):
        """Generate and upload exploration report"""
        
        print(f"\n[ANALYSIS] Generating exploration report...")
        
        # Calculate key metrics
        churn_counts = df['Churn'].value_counts()
        churn_props = df['Churn'].value_counts(normalize=True)
        overall_churn_rate = churn_props.get('Yes', 0)
        
        # Generate comprehensive report
        report = {
            'exploration_metadata': {
                'generated_at': datetime.now().isoformat(),
                'instance_type': os.environ.get('EC2_INSTANCE_TYPE', 'unknown'),
                'dataset_source': f"s3://{self.bucket_name}/{self.config['data']['raw_data_path']}",
                'script_version': '1.0.0'
            },
            'dataset_overview': {
                'total_rows': int(df.shape[0]),
                'total_columns': int(df.shape[1]),
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
            },
            'churn_analysis': {
                'overall_churn_rate': float(overall_churn_rate),
                'churn_counts': {
                    'no_churn': int(churn_counts.get('No', 0)),
                    'churn': int(churn_counts.get('Yes', 0))
                },
                'churn_percentages': {
                    'no_churn': float(churn_props.get('No', 0)),
                    'churn': float(churn_props.get('Yes', 0))
                }
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
        }
        
        # Add categorical insights
        categorical_insights = {}
        important_cats = ['Contract', 'InternetService', 'PaymentMethod', 'gender', 'SeniorCitizen']
        
        for col in important_cats:
            if col in df.columns:
                crosstab = pd.crosstab(df[col], df['Churn'], normalize='index')
                if 'Yes' in crosstab.columns:
                    churn_rates = crosstab['Yes'].sort_values(ascending=False)
                    categorical_insights[col] = {
                        'highest_churn_category': str(churn_rates.index[0]),
                        'highest_churn_rate': float(churn_rates.iloc[0]),
                        'lowest_churn_category': str(churn_rates.index[-1]),
                        'lowest_churn_rate': float(churn_rates.iloc[-1]),
                        'all_rates': {str(k): float(v) for k, v in churn_rates.to_dict().items()}
                    }
        
        report['categorical_insights'] = categorical_insights
        
        # Add numeric insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            numeric_insights = {}
            for col in numeric_cols:
                if col != 'customerID':  # Skip ID columns
                    numeric_insights[col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
            report['numeric_insights'] = numeric_insights
        
        # Save and upload report
        report_path = self.temp_dir / 'exploration_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Upload to S3
        s3_key = 'outputs/reports/exploration_report.json'
        metadata = {
            'content-type': 'application/json',
            'generated-by': 'ec2-exploration-script',
            'dataset-rows': str(df.shape[0]),
            'churn-rate': f"{overall_churn_rate:.3f}"
        }
        
        try:
            self.s3_client.upload_file(
                str(report_path),
                self.bucket_name,
                s3_key,
                ExtraArgs={'Metadata': metadata, 'ContentType': 'application/json'}
            )
            print(f"[UPLOADED] Exploration report: s3://{self.bucket_name}/{s3_key}")
        except ClientError as e:
            print(f"[ERROR] Failed to upload report: {e}")
        
        return report
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        
        print(f"\n[VISUALIZATION] Creating plots...")
        
        # Data preparation
        df_plot = df.copy()
        
        # Fix TotalCharges if it's object type
        if df_plot['TotalCharges'].dtype == 'object':
            df_plot['TotalCharges'] = pd.to_numeric(df_plot['TotalCharges'], errors='coerce')
            print(f"[DATA PREP] Converted TotalCharges to numeric")
        
        # Get churn distribution
        churn_counts = df['Churn'].value_counts()
        
        # Create main visualization dashboard - expanded to 4x3 grid
        fig, axes = self.plt.subplots(4, 3, figsize=(20, 20))
        fig.suptitle('Telco Customer Churn Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Churn Distribution (Pie Chart)
        axes[0,0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                      colors=['lightblue', 'lightcoral'], startangle=90)
        axes[0,0].set_title('Customer Churn Distribution', fontweight='bold', fontsize=12)
        
        # 2. Churn Distribution (Bar Chart)
        churn_counts.plot(kind='bar', ax=axes[0,1], color=['lightblue', 'lightcoral'])
        axes[0,1].set_title('Churn Count by Category', fontweight='bold')
        axes[0,1].set_xlabel('Churn Status')
        axes[0,1].set_ylabel('Number of Customers')
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # 3. Monthly Charges Distribution
        axes[0,2].hist(df_plot['MonthlyCharges'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,2].set_title('Monthly Charges Distribution', fontweight='bold')
        axes[0,2].set_xlabel('Monthly Charges ($)')
        axes[0,2].set_ylabel('Frequency')
        
        # 4. Monthly Charges by Churn (Box Plot)
        self.sns.boxplot(data=df_plot, x='Churn', y='MonthlyCharges', ax=axes[1,0])
        axes[1,0].set_title('Monthly Charges by Churn Status', fontweight='bold')
        axes[1,0].set_xlabel('Churn Status')
        axes[1,0].set_ylabel('Monthly Charges ($)')
        
        # 5. Tenure Distribution
        axes[1,1].hist(df_plot['tenure'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,1].set_title('Customer Tenure Distribution', fontweight='bold')
        axes[1,1].set_xlabel('Tenure (months)')
        axes[1,1].set_ylabel('Frequency')
        
        # 6. Tenure by Churn (Box Plot)
        self.sns.boxplot(data=df_plot, x='Churn', y='tenure', ax=axes[1,2])
        axes[1,2].set_title('Tenure by Churn Status', fontweight='bold')
        axes[1,2].set_xlabel('Churn Status')
        axes[1,2].set_ylabel('Tenure (months)')
        
        # 7. Contract Type vs Churn
        if 'Contract' in df.columns:
            contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index')
            contract_churn.plot(kind='bar', ax=axes[2,0], color=['lightblue', 'lightcoral'])
            axes[2,0].set_title('Churn Rate by Contract Type', fontweight='bold')
            axes[2,0].set_xlabel('Contract Type')
            axes[2,0].set_ylabel('Proportion')
            axes[2,0].tick_params(axis='x', rotation=45)
            axes[2,0].legend(['No Churn', 'Churn'])
        
        # 8. Internet Service vs Churn
        if 'InternetService' in df.columns:
            internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index')
            internet_churn.plot(kind='bar', ax=axes[2,1], color=['lightblue', 'lightcoral'])
            axes[2,1].set_title('Churn Rate by Internet Service', fontweight='bold')
            axes[2,1].set_xlabel('Internet Service Type')
            axes[2,1].set_ylabel('Proportion')
            axes[2,1].tick_params(axis='x', rotation=45)
            axes[2,1].legend(['No Churn', 'Churn'])
        
        # 9. Payment Method vs Churn
        if 'PaymentMethod' in df.columns:
            payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')
            payment_churn.plot(kind='bar', ax=axes[2,2], color=['lightblue', 'lightcoral'])
            axes[2,2].set_title('Churn Rate by Payment Method', fontweight='bold')
            axes[2,2].set_xlabel('Payment Method')
            axes[2,2].set_ylabel('Proportion')
            axes[2,2].tick_params(axis='x', rotation=45)
            axes[2,2].legend(['No Churn', 'Churn'])
        
        # 10. Gender vs Churn
        if 'gender' in df.columns:
            gender_churn = pd.crosstab(df['gender'], df['Churn'], normalize='index')
            bars = gender_churn.plot(kind='bar', ax=axes[3,0], color=['lightblue', 'lightcoral'])
            axes[3,0].set_title('Churn Rate by Gender', fontweight='bold')
            axes[3,0].set_xlabel('Gender')
            axes[3,0].set_ylabel('Proportion')
            axes[3,0].tick_params(axis='x', rotation=0)
            axes[3,0].legend(['No Churn', 'Churn'])
        
        # 11. Senior Citizen vs Churn
        if 'SeniorCitizen' in df.columns:
            # Convert SeniorCitizen to readable labels
            df_plot['SeniorCitizenLabel'] = df_plot['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'})
            senior_churn = pd.crosstab(df_plot['SeniorCitizenLabel'], df['Churn'], normalize='index')
            bars = senior_churn.plot(kind='bar', ax=axes[3,1], color=['lightblue', 'lightcoral'])
            axes[3,1].set_title('Churn Rate by Senior Citizen Status', fontweight='bold')
            axes[3,1].set_xlabel('Customer Type')
            axes[3,1].set_ylabel('Proportion')
            axes[3,1].tick_params(axis='x', rotation=0)
            axes[3,1].legend(['No Churn', 'Churn'])
        
        # 12. Combined Demographics Analysis
        if 'gender' in df.columns and 'SeniorCitizen' in df.columns:
            # Create combined demographic categories
            df_plot['Demographics'] = df_plot['gender'] + ' ' + df_plot['SeniorCitizenLabel']
            demo_churn = pd.crosstab(df_plot['Demographics'], df['Churn'], normalize='index')
            bars = demo_churn.plot(kind='bar', ax=axes[3,2], color=['lightblue', 'lightcoral'])
            axes[3,2].set_title('Churn Rate by Gender & Senior Status', fontweight='bold')
            axes[3,2].set_xlabel('Demographics')
            axes[3,2].set_ylabel('Proportion')
            axes[3,2].tick_params(axis='x', rotation=30, labelsize=9)  # Smaller rotation, smaller font
            axes[3,2].legend(['No Churn', 'Churn'])
        
        # Adjust layout with proper spacing for title
        self.plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave 4% space at top for title
        
        # Save main dashboard
        main_plot_path = self.temp_dir / 'telco_churn_dashboard.png'
        self.plt.savefig(main_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        self.plt.close()
        
        # Upload main dashboard
        s3_key = 'outputs/plots/telco_churn_dashboard.png'
        metadata = {
            'plot-type': 'dashboard',
            'generated-by': 'ec2-exploration',
            'plot-count': '12'
        }
        self.upload_plot_to_s3(main_plot_path, s3_key, metadata)
        
        # Create correlation heatmap
        numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = self.plt.subplots(1, 1, figsize=(12, 10))
            
            corr_matrix = df_plot[numeric_cols].corr()
            
            self.sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, ax=ax, cbar_kws={"shrink": .8})
            ax.set_title('Correlation Matrix of Numeric Variables', fontweight='bold', fontsize=14)
            
            self.plt.tight_layout()
            
            # Save heatmap
            heatmap_path = self.temp_dir / 'correlation_heatmap.png'
            self.plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
            self.plt.close()
            
            # Upload heatmap
            s3_key = 'outputs/plots/correlation_heatmap.png'
            metadata = {
                'plot-type': 'heatmap',
                'generated-by': 'ec2-exploration',
                'variables-count': str(len(numeric_cols))
            }
            self.upload_plot_to_s3(heatmap_path, s3_key, metadata)
        
        print(f"[SUCCESS] All visualizations created and uploaded")
    
    def run_exploration(self):
        """Main exploration workflow"""
        
        try:
            # Step 1: Download data from S3
            data_path = self.download_data_from_s3()
            
            # Step 2: Load and validate data
            print(f"\n[DATA] Loading dataset...")
            df = pd.read_csv(data_path)
            print(f"[SUCCESS] Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Step 3: Generate exploration report
            report = self.create_exploration_report(df)
            
            # Step 4: Create visualizations
            self.create_visualizations(df)
            
            # Step 5: Print summary
            self.print_summary(report)
            
            print(f"\n[SUCCESS] EC2/S3 data exploration completed successfully!")
            
        except Exception as e:
            print(f"[ERROR] Exploration failed: {e}")
            raise
        finally:
            # Cleanup temp directory
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
                print(f"[CLEANUP] Temporary files removed")
    
    def print_summary(self, report):
        """Print exploration summary"""
        
        print(f"\n" + "="*60)
        print(f"[SUMMARY] EXPLORATION RESULTS")
        print(f"="*60)
        
        print(f"[DATASET] {report['dataset_overview']['total_rows']:,} customers, "
              f"{report['dataset_overview']['total_columns']} features")
        
        churn_rate = report['churn_analysis']['overall_churn_rate']
        print(f"[CHURN RATE] {churn_rate:.1%}")
        
        print(f"\n[KEY DEMOGRAPHIC INSIGHTS]")
        
        # Prioritize demographic insights
        demo_categories = ['gender', 'SeniorCitizen']
        for category in demo_categories:
            if category in report['categorical_insights']:
                insights = report['categorical_insights'][category]
                highest = insights['highest_churn_category']
                highest_rate = insights['highest_churn_rate']
                lowest = insights['lowest_churn_category']
                lowest_rate = insights['lowest_churn_rate']
                
                print(f"   {category}: {highest} ({highest_rate:.1%}) vs {lowest} ({lowest_rate:.1%})")
        
        print(f"\n[SERVICE INSIGHTS]")
        
        service_categories = ['Contract', 'InternetService', 'PaymentMethod']
        for category in service_categories:
            if category in report['categorical_insights']:
                insights = report['categorical_insights'][category]
                highest = insights['highest_churn_category']
                highest_rate = insights['highest_churn_rate']
                lowest = insights['lowest_churn_category']
                lowest_rate = insights['lowest_churn_rate']
                
                print(f"   {category}: {highest} ({highest_rate:.1%}) vs {lowest} ({lowest_rate:.1%})")
        
        print(f"\n[S3 OUTPUTS]")
        print(f"   Dashboard: s3://{self.bucket_name}/outputs/plots/telco_churn_dashboard.png")
        print(f"   Heatmap: s3://{self.bucket_name}/outputs/plots/correlation_heatmap.png")
        print(f"   Report: s3://{self.bucket_name}/outputs/reports/exploration_report.json")

def main():
    """Main function for command line execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Telco Churn Data Exploration on EC2 with S3')
    parser.add_argument('--config', type=str, help='Path to AWS config file')
    parser.add_argument('--bucket', type=str, help='S3 bucket name (overrides config)')
    
    args = parser.parse_args()
    
    try:
        # Initialize explorer
        explorer = TelcoChurnS3Explorer(config_path=args.config)
        
        # Override bucket if specified
        if args.bucket:
            explorer.bucket_name = args.bucket
            print(f"[OVERRIDE] Using bucket: {args.bucket}")
        
        # Run exploration
        explorer.run_exploration()
        
        print(f"\n[COMPLETE] Data exploration finished successfully!")
        print(f"[NEXT STEP] Ready for preprocessing and model training!")
        
    except Exception as e:
        print(f"[ERROR] Exploration failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
