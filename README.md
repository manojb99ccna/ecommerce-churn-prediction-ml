# 🛒 Ecommerce Churn Prediction (ML)

Predict whether an e-commerce customer will churn or stay using a simple, reproducible ML pipeline built with Python, Pandas, scikit-learn, and FastAPI.

This repo includes end-to-end steps: synthetic data generation → preprocessing → model training → local prediction → REST API.

---

## 📁 Project Structure

```
ecommerce-churn-prediction-ml/
├── data/
│   ├── raw/                      # Synthetic raw CSV dataset
│   └── processed/                # Preprocessed (encoded + scaled) dataset
├── model/                        # Artifacts saved after preprocessing/training
│   ├── model.pkl                 # Trained RandomForest model
│   ├── preprocess_pipeline.pkl   # ColumnTransformer with scalers/encoders
│   └── columns.json              # Final feature column names
├── src/
│   ├── generate_dataset.py       # Synthetic dataset generator
│   ├── preprocess.py             # Build preprocessing pipeline + processed CSV
│   ├── 4_train_model.py          # Train RandomForest and save artifacts
│   ├── 5_predict.py              # Local test prediction script
│   └── 6_api.py                  # FastAPI app exposing /predict endpoint
├── requirements.txt              # Python dependencies
└── README.md                     # This document
```

---

## ✅ Requirements

- `Python 3.9+`
- `pip`
- Optional: `virtualenv` or the built‑in `venv`

---

## ⚙️ Setup

1) Create and activate a virtual environment

```
python -m venv venv
.\venv\Scripts\activate   # Windows PowerShell
# or
source venv/bin/activate     # macOS/Linux
```

2) Install dependencies

```
pip install -r requirements.txt
```

---

## 🚀 Workflow (End-to-End)

1) Generate a synthetic dataset (optional if you already have `data/raw/churn_full.csv`)

```
python src/generate_dataset.py
```

- Output: `data/raw/churn_full.csv`
- To customize rows or mode, edit the defaults inside `generate_dataset.py` (e.g., `rows`, `mode`).

2) Preprocess the data (build pipeline, encode/canonicalize columns)

```
python src/preprocess.py
```

- Reads: `data/raw/churn_full.csv`
- Writes: `data/processed/churn_full_processed.csv`, `model/preprocess_pipeline.pkl`, `model/columns.json`
- Behavior:
  - Renames `churned` → `churn` if needed
  - Drops `customer_id`
  - Scales numeric features and one‑hot encodes categorical features

3) Train the model (RandomForestClassifier)

```
python src/4_train_model.py
```

- Reads: `data/processed/churn_full_processed.csv`
- Writes: `model/model.pkl`, updates `model/columns.json`
- Prints accuracy/precision/recall/F1 on the holdout split

4) Test a local prediction

```
python src/5_predict.py
```

- Uses the saved `model.pkl` + `preprocess_pipeline.pkl`
- Edit the `test_customer` dict inside the script to try other inputs

5) Run the API server (FastAPI + Uvicorn)

```
uvicorn --app-dir src 6_api:app --reload
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- Root health: `http://127.0.0.1:8000/`

Note: Using `--app-dir src` avoids Python module path issues when `src` isn’t a package.

---

## 🧠 Model & Features

- Algorithm: `RandomForestClassifier` with `n_estimators=100`, `random_state=42`
- Numeric features: `age`, `account_age_days`, `last_login_days`, `total_spent`, `orders_count`, `support_tickets`
- Categorical features: `gender`, `country`, `membership`, `currency`
- Target: `churn` (0 = stay, 1 = churn)
- Preprocessing: `ColumnTransformer(OneHotEncoder + StandardScaler)`

Artifacts saved in `model/` after preprocessing/training:

- `preprocess_pipeline.pkl` — fitted transformers
- `columns.json` — final feature names after encoding
- `model.pkl` — trained classifier

---

## 📦 API Usage

Endpoint: `POST /predict`

Example request JSON:

```
{
  "age": 32,
  "account_age_days": 365,
  "last_login_days": 300,
  "total_spent": 1200,
  "orders_count": 25,
  "support_tickets": 1,
  "gender": "Male",
  "country": "Canada",
  "membership": "Silver",
  "currency": "CAD"
}
```

Example response:

```
{
  "prediction": 1,
  "probability": 0.88,
  "message": "⚠️ Customer is likely to leave"
}
```

Interpretation:

- `prediction`: `0` = stay, `1` = churn
- `probability`: model’s confidence for the churn class
- `message`: human‑friendly summary

---

## 🔍 Data Notes

- Synthetic generation uses sensible heuristics:
  - Inactivity (`last_login_days > 30`), low orders/spend, and many support tickets increase churn probability.
  - Membership tiers affect churn probability slightly.
- Reproducibility: fixed seeds in dataset generation.

---

## 🛠️ Troubleshooting

- “Module not found” when starting Uvicorn:
  - Use `uvicorn --app-dir src 6_api:app --reload` from the project root.
- API returns preprocessing shape errors:
  - Ensure you ran `python src/preprocess.py` and `python src/4_train_model.py` so `model/columns.json` matches the pipeline.
- CSV not found:
  - Run `python src/generate_dataset.py` or place your dataset at `data/raw/churn_full.csv`.

---

## 🧭 Next Steps

- Try different models or hyperparameters
- Add proper CLI arguments (argparse) to scripts
- Log metrics and track experiments
- Persist API logs or add authentication

---

## 🙌 Acknowledgements

- Built with Pandas, scikit‑learn, and FastAPI.

Enjoy exploring and extending the churn prediction pipeline!