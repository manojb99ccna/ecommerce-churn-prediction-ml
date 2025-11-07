# src/6_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

app = FastAPI(title="Ecommerce Churn Prediction API")

# ====== Load Model & Preprocessing Pipeline ======
model = joblib.load("model/model.pkl")
pipeline = joblib.load("model/preprocess_pipeline.pkl")
with open("model/columns.json", "r") as f:
    feature_columns = json.load(f)

# ====== Define Request Schema ======
class CustomerData(BaseModel):
    age: float
    account_age_days: float
    last_login_days: float
    total_spent: float
    orders_count: float
    support_tickets: float
    gender: str
    country: str
    membership: str
    currency: str

# ====== API Root ======
@app.get("/")
def home():
    return {"message": "Welcome to Ecommerce Churn Prediction API"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])

        # Apply preprocessing
        processed = pipeline.transform(df)
        X_processed = pd.DataFrame(processed, columns=feature_columns)

        # Predict
        prediction = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])


        result = (
            "⚠️ Customer is likely to leave"
            if prediction == 1
            else "✅ Customer is likely to stay"
        )

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 2),
            "message": result,
        }

    except Exception as e:
        # Print error to console + return to API
        print("❌ ERROR:", str(e))
        return {"error": str(e)} 