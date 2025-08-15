# 🚀 Telco Customer Churn Prediction System
*End-to-End Machine Learning Pipeline with AWS Cloud Deployment*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-orange.svg)](https://aws.amazon.com/)
[![Flask](https://img.shields.io/badge/Flask-API-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Dataset
This project uses the Telco Customer Churn dataset, originally from:
- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **License**: Public Domain
- **Size**: 7,043 customers, 21 features
- **Included**: Complete dataset available in `data/raw/`

## 📊 **Project Overview**

A complete machine learning system that predicts customer churn for telecommunications companies, featuring automated data preprocessing, model training, and real-time prediction API deployed on AWS cloud infrastructure.

### **🎯 Key Results**
- **80.8% Model Accuracy** with Logistic Regression
- **Production-Ready API** with interactive web interface
- **AWS Cloud Deployment** with cost-optimized infrastructure
- **Advanced Feature Engineering** solving complex categorical encoding challenges

### **🔧 Technologies Used**
- **ML/Data Science**: Python, scikit-learn, pandas, numpy
- **Cloud Infrastructure**: AWS S3, EC2
- **API Development**: Flask, RESTful API design
- **Frontend**: HTML, CSS, JavaScript
- **DevOps**: Git, logging, error handling

---

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  AWS S3 Storage │───▶│ Processing Pipeline│
│ (Telco Dataset)│    │   Raw Data      │    │   Data Cleaning   │
└─────────────────┘    └─────────────────┘    │ Feature Engineering│
                                              │   Encoding        │
                                              └─────────┬─────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐            │
│   Inference     │◀───│  Trained Models │◀───────────┘
│   Flask API     │    │    (S3 Storage) │    
│                 │    │                 │    ┌─────────────────┐
│ • RESTful API   │    │ • Best Model    │    │  Training Pipeline│
│ • Web Interface │    │ • Encoders      │    │                 │
│ • Real-time     │    │ • Scalers       │    │ • Multiple Models│
│   Predictions   │    │ • Metadata      │    │ • Hyperparameter │
└─────────────────┘    └─────────────────┘    │   Optimization   │
                                              │ • Cross-validation│
        ▲                                     └─────────────────┘
        │
┌─────────────────┐
│   AWS EC2       │
│  (t2.micro)     │
└─────────────────┘
```

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- AWS Account with S3 and EC2 access
- AWS CLI configured

### **Installation**
```bash
# Clone repository
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

### **Configuration**
1. Copy `config/aws_config_template.json` to `config/aws_config.json`
2. Update with your AWS S3 bucket name and region
3. Upload your dataset to S3

### **Usage**
```bash
# 1. Run data preprocessing
python scripts/s3_preprocessing_pipeline.py

# 2. Train models
python scripts/micro_training_pipeline.py

# 3. Start API server
python scripts/API.py

# 4. Access web interface
# Open browser to http://localhost:5000
```

---

## 📈 **Model Performance**

| Model | ROC AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.8247** | **80.8%** | **0.7156** | **0.6892** | **0.7021** |
| Random Forest | 0.8156 | 79.3% | 0.6945 | 0.6734 | 0.6838 |
| Decision Tree | 0.7892 | 76.8% | 0.6523 | 0.6245 | 0.6381 |
| Naive Bayes | 0.7634 | 74.2% | 0.6198 | 0.5987 | 0.6091 |

---

## 🌐 **API Documentation**

### **Endpoints**
- `GET /` - Interactive web interface
- `GET /health` - API health status
- `GET /model_info` - Model performance metrics
- `POST /predict` - Single customer prediction
- `POST /batch_predict` - Batch predictions

### **Example Usage**
```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "tenure": 24,
    "Contract": "Two year",
    "MonthlyCharges": 65.50,
    "TotalCharges": "1572.00"
  }'

# Response
{
  "success": true,
  "prediction": "No",
  "churn_probability": 0.23,
  "confidence": 0.77
}
```

---

## 🔧 **Key Technical Challenges Solved**

### **1. Categorical Feature Alignment**
**Problem**: Feature names between training and inference didn't match due to categorical encoding differences.

**Solution**: Implemented reverse engineering approach to extract exact categorical values from trained model features, ensuring consistent one-hot encoding.

### **2. Memory Optimization for t2.micro**
**Problem**: Limited RAM (1GB) on cost-effective EC2 instances.

**Solution**: Created memory-efficient training pipeline with sequential processing, optimized data types, and garbage collection.

### **3. Production-Ready Error Handling**
**Problem**: Robust error handling for production deployment.

**Solution**: Comprehensive logging, graceful degradation, input validation, and detailed debugging tools.

---

## 📁 **Project Structure**
```
telco-churn-prediction/
├── scripts/
│   ├── s3_preprocessing_pipeline.py    # Data preprocessing
│   ├── micro_training_pipeline.py      # Model training
│   └── API.py                          # Flask API server
├── config/
│   ├── aws_config_template.json        # Configuration template
│   └── aws_config.json                 # Your AWS config (gitignored)
├── docs/
│   └── technical_details.md            # Detailed documentation
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

---

## 🚀 **AWS Deployment**

### **Infrastructure Setup**
1. **S3 Bucket**: Store data, models, and artifacts
2. **EC2 Instance**: t2.micro for cost-effective hosting
3. **Security Groups**: Configure appropriate access rules

### **Deployment Commands**
```bash
# Launch EC2 instance
aws ec2 run-instances --image-id ami-0abcdef1234567890 --instance-type t2.micro

# Deploy application
scp -i keypair.pem -r . ec2-user@<instance-ip>:~/telco-churn/

# Start production server
nohup python API.py > api.log 2>&1 &
```

---

## 📊 **Business Impact**

### **ROI Analysis**
- **Customer Acquisition Cost**: $200-500 (industry average)
- **Retention Campaign Cost**: $20-50 per customer
- **Model Accuracy**: 80.8% precision in identifying at-risk customers
- **Potential Savings**: $150-450 per correctly identified customer

### **Customer Insights**
- **High-Risk Profile**: Month-to-month contracts, electronic check payments, short tenure
- **Low-Risk Profile**: Long-term contracts, automatic payments, multiple services
- **Key Predictors**: Contract type, tenure, payment method, monthly charges

---

## 🔮 **Future Enhancements**

- [ ] **Advanced Models**: XGBoost, Neural Networks
- [ ] **Real-time Streaming**: Kafka/Kinesis integration
- [ ] **Model Monitoring**: Drift detection and automated retraining
- [ ] **Container Deployment**: Docker + Kubernetes
- [ ] **CI/CD Pipeline**: Automated testing and deployment

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 **Author**

**Your Name**
- GitHub: [@yshafeen07](https://github.com/shafeen07)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/shafeen-ahmed-baa39830/)
- Email: shafeenahmed07.gmail@example.com

---

## 🙏 **Acknowledgments**

- Dataset: Telco Customer Churn from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Cloud Platform: Amazon Web Services
- ML Libraries: scikit-learn, pandas, numpy

---

*⭐ Star this repository if you found it helpful!*