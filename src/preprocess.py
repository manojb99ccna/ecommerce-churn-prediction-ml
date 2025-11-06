import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json

# ---------- CONFIG ----------
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_PATH = "model/"

# Hardcoded file name
file_name = "churn_small.csv"

numeric_cols = [
    "age",
    "account_age_days",
    "last_login_days",
    "total_spent",
    "orders_count",
    "support_tickets"
]

categorical_cols = [
    "gender",
    "country",
    "membership",
    "currency"
]

target_col = "churn"
# --------------------------------

def preprocess():
    file_path = os.path.join(RAW_DATA_PATH, file_name)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Loading dataset: {file_path}")
    df = pd.read_csv(file_path)

    print(f"✅ Loaded data: {df.shape}")

    # Remove missing values
    df = df.dropna()

    # Separate target (if exists)
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    X_processed = pipeline.fit_transform(X)

    # Get final feature names
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    ohe_features = ohe.get_feature_names_out(categorical_cols)

    final_columns = numeric_cols + list(ohe_features)

    # Save processed CSV
    processed_df = pd.DataFrame(X_processed, columns=final_columns)
    if y is not None:
        processed_df[target_col] = y.values

    processed_file = file_name.replace(".csv", "_processed.csv")
    processed_path = os.path.join(PROCESSED_DATA_PATH, processed_file)
    processed_df.to_csv(processed_path, index=False)

    print(f"✅ Processed data saved: {processed_path}")

    # Save scaler/encoder pipeline
    joblib.dump(pipeline, os.path.join(MODEL_PATH, "preprocess_pipeline.pkl"))
    print(f"✅ Saved preprocessing pipeline → model/preprocess_pipeline.pkl")

    # Save final columns JSON
    with open(os.path.join(MODEL_PATH, "columns.json"), "w") as f:
        json.dump(final_columns, f, indent=2)
    print(f"✅ Saved feature column list → model/columns.json")

    print("\n🎉 Preprocessing complete!")

if __name__ == "__main__":
    preprocess()
