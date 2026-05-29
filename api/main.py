from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np , os

app = FastAPI(title="Customer Churn Prediction API")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = pickle.load(open(os.path.join(base_dir, "model", "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(base_dir,"model", "scaler.pkl"), "rb"))

class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int 
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    Techsupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return{"message": "Customer Churn Prediction API is running!"}

@app.post("/predict")
def predict(customer: Customer):
    data =np.array([[
        customer.gender, customer.SeniorCitizen, customer.Partner, customer.Dependents, customer.tenure,
        customer.PhoneService, customer.MultipleLines, customer.InternetService, customer.OnlineSecurity,
        customer.OnlineBackup, customer.DeviceProtection, customer.Techsupport, customer.StreamingTV,
        customer.StreamingMovies, customer.Contract, customer.PaperlessBilling, customer.PaymentMethod, 
        customer.MonthlyCharges, customer.TotalCharges
    ]])

    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability)*100,2)
    }