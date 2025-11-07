import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

# === Load processed data ===
df = pd.read_csv("data/processed/churn_full_processed.csv")
print(f"✅ Loaded processed data: {df.shape}")

# === Detect target column ===
target_col = "churn"
if target_col not in df.columns:
    raise ValueError("❌ Target column 'churn' not found in dataset!")

# === Split features and target ===
X = df.drop(target_col, axis=1)
y = df[target_col]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model training ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n📊 Model Evaluation:")
print(f"✅ Accuracy : {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall   : {recall:.2f}")
print(f"✅ F1 Score : {f1:.2f}")

# === Save model and metadata ===
joblib.dump(model, "model/model.pkl")
with open("model/columns.json", "w") as f:
    json.dump(list(X.columns), f)

print("\n✅ Model and feature columns saved!")
