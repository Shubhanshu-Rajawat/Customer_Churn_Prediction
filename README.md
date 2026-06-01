# Customer_Churn_Prediction
### End-to-End ML Pipeline with FastAPI Deployment

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> A complete end-to-end Machine Learning pipeline that predicts whether a telecom customer will churn (leave) or not. Built on the Telco Customer Churn dataset with 7,043 records, deployed as a production-ready FastAPI REST API with Swagger documentation.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Sample API Request](#sample-api-request)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

This project builds a **binary churn classifier** for telecom customers. Given a customer's account details and service usage, the model predicts:
- **Yes** → Customer will likely leave (churn)
- **No** → Customer will likely stay

**Key Highlights:**
- Real-world dataset with 7,043 telecom customers
- Complete preprocessing pipeline (encoding, scaling, NaN handling)
- Two ML models compared — Logistic Regression vs Random Forest
- Production-ready FastAPI REST API with auto-generated Swagger UI
- Modular, clean code structure

---

## 🗃️ Dataset

| Property | Details |
|----------|---------|
| Name | Telco Customer Churn |
| Source | [Kaggle - blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Originally by | IBM Sample Data |
| Total Records | 7,043 customers |
| Total Features | 21 columns |
| Target Column | Churn (Yes/No) |
| Churned | 1,869 customers (26.5%) |
| Not Churned | 5,174 customers (73.5%) |

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│
├── model/
│   ├── model.pkl          # Trained Random Forest model
│   └── scaler.pkl         # Fitted StandardScaler
│
├── src/
│   ├── preprocess.py      # Data cleaning & preprocessing pipeline
│   └── train.py           # Model training & evaluation
│
├── api/
│   └── main.py            # FastAPI REST API
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip
- Git

### Step 1 — Clone Repository
```bash
git clone https://github.com/Shubhanshu-Rajawat/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download Dataset
Download from Kaggle and place in `data/` folder.

---

## 🚀 How to Run

### Step 1 — Preprocess Data
```bash
python src/preprocess.py
```
Output:
```
Preprocessing Done
Train Size: (5634, 19)
Test  Size: (1409, 19)
```

### Step 2 — Train Model
```bash
python src/train.py
```
Output:
```
✅ Logistic Regression Accuracy: 79.84 %  ROC-AUC: 0.84
✅ Random Forest Accuracy: 78.64 %        ROC-AUC: 0.81
Model Saved!
```

### Step 3 — Start API
```bash
uvicorn api.main:app --reload
```

### Step 4 — Open Swagger UI
```
http://localhost:8000/docs
```

---

## 🔬 ML Pipeline

### Preprocessing Steps
1. Drop `customerID` (not useful)
2. Fix `TotalCharges` (string → float, fill NaN with median)
3. Encode `Churn` column (Yes→1, No→0)
4. Label encode all categorical columns (15 columns)
5. Train/Test split (80/20, stratified)
6. StandardScaler normalization

### Models Trained
- **Logistic Regression** — Baseline model
- **Random Forest** — Ensemble model (100 trees)

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 79.84% | 0.84 |
| Random Forest | 78.64% | 0.81 |

### Classification Report (Random Forest)
```
              precision    recall  f1-score   support
           0       0.83      0.90      0.86      1035
           1       0.63      0.48      0.54       374
    accuracy                           0.79      1409
```

---

## 🌐 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict churn |
| GET | `/docs` | Swagger UI |

### Input Fields (19 features)

| Field | Type | Example |
|-------|------|---------|
| gender | int | 0=Female, 1=Male |
| SeniorCitizen | int | 0 or 1 |
| tenure | int | 12 (months) |
| MonthlyCharges | float | 65.5 |
| TotalCharges | float | 786.0 |
| Contract | int | 0=Monthly, 1=1yr, 2=2yr |
| ... | ... | ... |

---

## 🧪 Sample API Request

```json
POST http://localhost:8000/predict

{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 12,
  "PhoneService": 1,
  "MultipleLines": 0,
  "InternetService": 1,
  "OnlineSecurity": 0,
  "OnlineBackup": 1,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "Contract": 0,
  "PaperlessBilling": 1,
  "PaymentMethod": 2,
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0
}
```

### Response
```json
{
  "churn_prediction": "No",
  "churn_probability": 41
}
```

---

## 🛠️ Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.14 |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| API Framework | FastAPI |
| API Server | Uvicorn |
| Data Validation | Pydantic |
| IDE | VS Code |
| Version Control | Git & GitHub |

---

## 🔮 Future Improvements

- [ ] Add XGBoost model for 87%+ accuracy
- [ ] Fix class imbalance with SMOTE
- [ ] Add hyperparameter tuning (GridSearchCV)
- [ ] Deploy on Railway/Render cloud
- [ ] Add Streamlit frontend UI
- [ ] Docker containerization

---

## 📝 Resume Description

> *Built an end-to-end ML pipeline for customer churn prediction on Telco dataset (7,043 records). Implemented preprocessing pipeline with label encoding and StandardScaler, trained Logistic Regression (79.84% accuracy, ROC-AUC: 0.84) and Random Forest models, and deployed predictions via FastAPI REST API with Swagger documentation.*

---

## 👨‍💻 Author

**Shubhanshu Rajawat**
- GitHub: [@Shubhanshu-Rajawat](https://github.com/Shubhanshu-Rajawat)

---

## 📄 License
MIT License

⭐ **Star this repo if you found it helpful!**
End-to-End ML Pipeline for Customer Churn Prediction using XgBoost &amp; FastAPI
