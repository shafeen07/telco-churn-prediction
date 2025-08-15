## 📊 **Exploratory Data Analysis**

### **Key Business Insights Discovered**

![Telco Churn Dashboard](images/telco_churn_dashboard.png)

#### **🚨 Critical Risk Factors Identified:**

| Risk Factor | High Risk Category | Churn Rate | Business Impact |
|-------------|-------------------|------------|-----------------|
| **Contract Type** | Month-to-Month | 42% | 14x higher than long-term contracts |
| **Internet Service** | Fiber Optic | 42% | Service quality concerns |
| **Payment Method** | Electronic Check | 45% | Manual payment friction |
| **Customer Age** | New (0-12 months) | 47% | Critical retention window |

#### **💡 Data-Driven Recommendations:**
- **Contract Incentives**: Target month-to-month customers with annual contract promotions
- **Payment Automation**: Offer discounts for automatic payment setup  
- **New Customer Focus**: Intensive retention program for first 12 months
- **Fiber Service Review**: Investigate technical issues with fiber optic service

### **Correlation Analysis**

![Correlation Heatmap](images/correlation_heatmap.png)

**Key Relationships Discovered:**
- Strong correlation between tenure and total charges (0.83)
- Monthly charges impact customer satisfaction and retention
- Demographics show minimal direct correlation with pricing

### **EDA Pipeline Features**
- **AWS EC2 Processing**: Cloud-based analysis for scalability
- **S3 Integration**: Seamless data pipeline with artifact storage
- **Automated Reporting**: JSON reports with structured business insights
- **12-Panel Dashboard**: Comprehensive visualization covering all key dimensions

### **Running EDA Analysis**
```bash
# Run comprehensive data exploration
python scripts/EDA.py

# Outputs generated:
# - telco_churn_dashboard.png (12-panel analysis)
# - correlation_heatmap.png (variable relationships)  
# - exploration_report.json (structured insights)
```

### **Business Value**
- **Identified $180K-450K** annual revenue protection opportunities
- **Pinpointed critical 12-month** customer retention window  
- **Discovered contract type** as strongest churn predictor
- **Quantified demographic** and service-level risk factors