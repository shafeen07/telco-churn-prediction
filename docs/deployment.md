# 🚀 AWS Deployment Guide

Complete step-by-step guide for deploying the Telco Customer Churn Prediction system on AWS infrastructure.

---

## 📋 **Prerequisites**

### **Required Accounts & Tools**
- ✅ **AWS Account** with billing enabled
- ✅ **AWS CLI** installed and configured
- ✅ **Python 3.8+** installed locally
- ✅ **Git** for repository management
- ✅ **SSH Key Pair** for EC2 access

### **Estimated Costs**
- **S3 Storage**: ~$0.05/month (for data and models)
- **EC2 t2.micro**: ~$8.50/month (or free tier eligible)
- **Data Transfer**: ~$0.01/month (minimal usage)
- **Total Monthly**: ~$8.60/month (or ~$0.06/month on free tier)

---

## 🔧 **AWS Infrastructure Setup**

### **Step 1: Configure AWS CLI**
```bash
# Install AWS CLI (if not already installed)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)

# Verify setup
aws sts get-caller-identity
```

### **Step 2: Create S3 Bucket**
```bash
# Create unique bucket name (replace with your choice)
BUCKET_NAME="telco-churn-prediction-$(date +%s)"
REGION="us-east-1"

# Create bucket
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Enable versioning (recommended)
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled

# Set up folder structure
aws s3api put-object --bucket $BUCKET_NAME --key data/raw/
aws s3api put-object --bucket $BUCKET_NAME --key data/processed/
aws s3api put-object --bucket $BUCKET_NAME --key models/trained/
aws s3api put-object --bucket $BUCKET_NAME --key models/encoders/
aws s3api put-object --bucket $BUCKET_NAME --key outputs/plots/
aws s3api put-object --bucket $BUCKET_NAME --key outputs/reports/

echo "✅ S3 bucket created: $BUCKET_NAME"
```

### **Step 3: Upload Dataset**
```bash
# Upload the telco dataset
aws s3 cp data/raw/telco_customer_churn.csv s3://$BUCKET_NAME/data/raw/

# Verify upload
aws s3 ls s3://$BUCKET_NAME/data/raw/
```

### **Step 4: Create IAM Role for EC2**
```bash
# Create trust policy
cat > ec2-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# Create IAM role
aws iam create-role \
    --role-name TelcoChurnEC2Role \
    --assume-role-policy-document file://ec2-trust-policy.json

# Create S3 access policy
cat > s3-access-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ]
        }
    ]
}
EOF

# Attach policy to role
aws iam put-role-policy \
    --role-name TelcoChurnEC2Role \
    --policy-name S3Access \
    --policy-document file://s3-access-policy.json

# Create instance profile
aws iam create-instance-profile --instance-profile-name TelcoChurnProfile
aws iam add-role-to-instance-profile \
    --instance-profile-name TelcoChurnProfile \
    --role-name TelcoChurnEC2Role

echo "✅ IAM role and instance profile created"
```

---

## 🖥️ **EC2 Instance Deployment**

### **Step 5: Launch EC2 Instance**
```bash
# Create key pair (if you don't have one)
aws ec2 create-key-pair \
    --key-name telco-churn-key \
    --query 'KeyMaterial' \
    --output text > telco-churn-key.pem

# Set permissions
chmod 400 telco-churn-key.pem

# Create security group
aws ec2 create-security-group \
    --group-name telco-churn-sg \
    --description "Security group for Telco Churn API"

# Get security group ID
SG_ID=$(aws ec2 describe-security-groups \
    --group-names telco-churn-sg \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0  # SSH access

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 5000 \
    --cidr 0.0.0.0/0  # Flask API access

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0abcdef1234567890 \  # Amazon Linux 2 AMI (update as needed)
    --count 1 \
    --instance-type t2.micro \
    --key-name telco-churn-key \
    --security-group-ids $SG_ID \
    --iam-instance-profile Name=TelcoChurnProfile \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=TelcoChurnAPI}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ EC2 instance launched: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ Instance ready at: $PUBLIC_IP"
```

### **Step 6: Connect and Setup Instance**
```bash
# Connect to instance
ssh -i telco-churn-key.pem ec2-user@$PUBLIC_IP

# Once connected, run these commands on the EC2 instance:
```

**On EC2 Instance:**
```bash
# Update system
sudo yum update -y

# Install Python 3.8+
sudo yum install python3 python3-pip git -y

# Install additional dependencies
sudo yum install gcc python3-devel -y

# Create project directory
mkdir ~/telco-churn
cd ~/telco-churn

# Clone repository
git clone https://github.com/yourusername/telco-churn-prediction.git .

# Install Python dependencies
pip3 install --user -r requirements.txt

# Verify AWS access (should work automatically with IAM role)
aws s3 ls

echo "✅ EC2 instance setup complete"
```

### **Step 7: Configure Application**
```bash
# Create AWS config file with your bucket name
cat > config/aws_config.json << EOF
{
    "aws": {
        "bucket_name": "$BUCKET_NAME",
        "region": "$REGION"
    },
    "preprocessing": {
        "test_size": 0.2,
        "val_size": 0.15,
        "random_state": 42
    },
    "data": {
        "raw_data_path": "data/raw/telco_customer_churn.csv"
    }
}
EOF

# Make scripts executable
chmod +x scripts/*.py
```

---

## 🔄 **Running the Complete Pipeline**

### **Step 8: Data Processing and Training**
```bash
# Run exploratory data analysis
python3 scripts/EDA.py
echo "✅ EDA completed - check S3 for visualizations"

# Run preprocessing pipeline
python3 scripts/s3_preprocessing_pipeline.py
echo "✅ Data preprocessing completed"

# Run training pipeline
python3 scripts/micro_training_pipeline.py
echo "✅ Model training completed"

# Verify artifacts in S3
aws s3 ls s3://$BUCKET_NAME/models/trained/
aws s3 ls s3://$BUCKET_NAME/outputs/plots/
```

### **Step 9: Deploy API**
```bash
# Start API server
nohup python3 scripts/API.py > api.log 2>&1 &

# Check if running
curl http://localhost:5000/health

# API is now available at: http://$PUBLIC_IP:5000
echo "✅ API deployed at: http://$PUBLIC_IP:5000"
```

---

## 🔍 **Verification and Testing**

### **Test Endpoints**
```bash
# Health check
curl http://$PUBLIC_IP:5000/health

# Model info
curl http://$PUBLIC_IP:5000/model_info

# Sample prediction
curl -X POST http://$PUBLIC_IP:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 65.50,
    "TotalCharges": "1572.00"
  }'
```

### **Web Interface Test**
1. **Open browser** to `http://$PUBLIC_IP:5000`
2. **Fill out form** with customer details
3. **Submit prediction** and verify results
4. **Test different scenarios** to validate model behavior

---

## 🛡️ **Security Best Practices**

### **Network Security**
```bash
# Restrict SSH access to your IP only
MY_IP=$(curl -s https://ipinfo.io/ip)
aws ec2 revoke-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr $MY_IP/32
```

### **S3 Security**
```bash
# Enable encryption on bucket
aws s3api put-bucket-encryption \
    --bucket $BUCKET_NAME \
    --server-side-encryption-configuration '{
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }
        ]
    }'

# Block public access
aws s3api put-public-access-block \
    --bucket $BUCKET_NAME \
    --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

---

## 📊 **Monitoring and Logs**

### **Application Logs**
```bash
# View API logs
tail -f ~/telco-churn/api.log

# Check system resources
htop
df -h
free -m
```

### **CloudWatch Integration (Optional)**
```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
sudo rpm -U ./amazon-cloudwatch-agent.rpm

# Configure basic monitoring
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

---

## 🔄 **Maintenance and Updates**

### **Application Updates**
```bash
# Update code from repository
cd ~/telco-churn
git pull origin main

# Restart API
pkill -f "python3 scripts/API.py"
nohup python3 scripts/API.py > api.log 2>&1 &
```

### **System Updates**
```bash
# Update system packages
sudo yum update -y

# Update Python packages
pip3 install --user --upgrade -r requirements.txt
```

### **Backup Strategy**
```bash
# S3 automatically handles data backups with versioning
# Create snapshot of EC2 instance for system backup
aws ec2 create-snapshot \
    --volume-id $(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].BlockDeviceMappings[0].Ebs.VolumeId' \
        --output text) \
    --description "TelcoChurn-$(date +%Y%m%d)"
```

---

## 🧹 **Cleanup (When Done)**

### **Stop Resources to Save Costs**
```bash
# Stop EC2 instance (preserves data, stops compute charges)
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Or terminate if permanently done (destroys instance)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID

# Delete S3 objects (optional - keeps bucket)
aws s3 rm s3://$BUCKET_NAME --recursive

# Delete S3 bucket (permanent)
aws s3 rb s3://$BUCKET_NAME
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **API Won't Start**
```bash
# Check Python dependencies
pip3 list | grep -E "(flask|boto3|pandas|sklearn)"

# Check AWS credentials
aws sts get-caller-identity

# Check S3 access
aws s3 ls s3://$BUCKET_NAME

# Check port availability
netstat -tlnp | grep :5000
```

#### **Model Loading Errors**
```bash
# Verify model files exist in S3
aws s3 ls s3://$BUCKET_NAME/models/trained/

# Check preprocessing artifacts
aws s3 ls s3://$BUCKET_NAME/models/encoders/

# Rerun training if needed
python3 scripts/micro_training_pipeline.py
```

#### **S3 Permission Errors**
```bash
# Check IAM role attachment
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].IamInstanceProfile'

# Test S3 access directly
aws s3 ls s3://$BUCKET_NAME --debug
```

#### **Memory Issues on t2.micro**
```bash
# Check memory usage
free -m

# Monitor during training
watch -n 1 'free -m && ps aux --sort=-%mem | head -10'

# If memory issues persist, upgrade to t2.small temporarily
```

### **Support Resources**
- **AWS Documentation**: [AWS EC2 User Guide](https://docs.aws.amazon.com/ec2/)
- **Flask Deployment**: [Flask Deployment Documentation](https://flask.palletsprojects.com/en/2.0.x/deploying/)
- **scikit-learn**: [Deployment Guidelines](https://scikit-learn.org/stable/model_persistence.html)

---

## ✅ **Deployment Checklist**

- [ ] **AWS CLI configured** with appropriate credentials
- [ ] **S3 bucket created** with proper folder structure
- [ ] **IAM roles configured** for EC2 S3 access
- [ ] **EC2 instance launched** with security group
- [ ] **Application code deployed** and dependencies installed
- [ ] **Dataset uploaded** to S3
- [ ] **EDA pipeline executed** successfully
- [ ] **Preprocessing completed** with artifacts in S3
- [ ] **Model training completed** with trained models
- [ ] **API deployed** and responding to requests
- [ ] **Web interface accessible** and functional
- [ ] **Security measures implemented** (restricted access)
- [ ] **Monitoring configured** (logs, health checks)
- [ ] **Backup strategy** in place

**🎉 Deployment Complete! Your Telco Churn Prediction system is now live on AWS.**