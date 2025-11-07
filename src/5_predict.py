import pandas as pd
import joblib
import json
import numpy as np

# ===== Load trained model & preprocessing info =====
model = joblib.load("model/model.pkl")
with open("model/columns.json", "r") as f:
    feature_columns = json.load(f)

pipeline = joblib.load("model/preprocess_pipeline.pkl")

# ===== Example customer data =====
# Provide values for the raw customer features (before preprocessing)
test_customer = {
    "age": 30,
    "account_age_days": 120,
    "last_login_days": 5,
    "total_spent": 500,
    "orders_count": 3,
    "support_tickets": 0,
    "gender": "Male",
    "country": "USA",
    "membership": "Silver",
    "currency": "USD"
}

# Convert dict to DataFrame
df_test = pd.DataFrame([test_customer])

# Apply preprocessing pipeline
X_test_processed = pipeline.transform(df_test)

# Ensure correct column order
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_columns)

# Predict churn
prediction = model.predict(X_test_processed_df)
result = "❌ Will NOT churn" if prediction[0] == 0 else "✅ Will churn"

print("🔍 Prediction result:")
print(result)
