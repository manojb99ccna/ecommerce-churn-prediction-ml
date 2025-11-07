import pandas as pd
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === File paths ===
RAW_PATH = "data/raw/churn_full.csv"  # change if needed
PROCESSED_PATH = "data/processed/churn_full_processed.csv"
PIPELINE_PATH = "model/preprocess_pipeline.pkl"
COLUMNS_PATH = "model/columns.json"

def preprocess():
    print(f"📂 Loading dataset: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"✅ Loaded data: {df.shape}")

    # === Normalize churn column ===
    if "churn" not in df.columns:
        if "churned" in df.columns:
            df = df.rename(columns={"churned": "churn"})
        else:
            raise ValueError("❌ Neither 'churn' nor 'churned' found in dataset")

    # === Drop ID column ===
    if "customer_id" in df.columns:
        df = df.drop("customer_id", axis=1)

    # === Split X and y ===
    X = df.drop("churn", axis=1)
    y = df["churn"].astype(int)

    # === Define columns ===
    num_cols = ["age", "account_age_days", "last_login_days", "total_spent", "orders_count", "support_tickets"]
    cat_cols = ["gender", "country", "membership", "currency"]

    # === Preprocessing pipeline ===
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    X_processed = pipeline.fit_transform(X)

    # === Get feature names ===
    cat_feature_names = pipeline.named_transformers_["cat"].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_feature_names)

    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_processed["churn"] = y.values

    # === Save processed data ===
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    df_processed.to_csv(PROCESSED_PATH, index=False)

    joblib.dump(pipeline, PIPELINE_PATH)
    with open(COLUMNS_PATH, "w") as f:
        json.dump(feature_names, f)

    print(f"✅ Processed data saved: {PROCESSED_PATH}")
    print(f"✅ Saved preprocessing pipeline → {PIPELINE_PATH}")
    print(f"✅ Saved feature column list → {COLUMNS_PATH}")
    print("\n🎉 Preprocessing complete!")

if __name__ == "__main__":
    preprocess()
