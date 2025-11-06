import os
import argparse
import pandas as pd
import numpy as np

# Currency conversion to USD (approx)
CURRENCY_MAP = {
    "USA": ("USD", 1.00),
    "Canada": ("CAD", 0.75),
    "UK": ("GBP", 1.25),
    "Germany": ("EUR", 1.08),
    "India": ("INR", 0.012),
    "Australia": ("AUD", 0.66),
    "Bangladesh": ("BDT", 0.0091),
    "Sri Lanka": ("LKR", 0.0033),
    "Pakistan": ("PKR", 0.0036)
}

GENDERS = ["Male", "Female", "Other"]
MEMBERSHIP = ["Free", "Silver", "Gold", "Platinum"]
COUNTRIES = list(CURRENCY_MAP.keys())


def generate_dataset(rows=1000, start_index=1, seed=42):
    np.random.seed(seed)

    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(start_index, start_index + rows)]

    ages = np.random.randint(18, 70, rows)
    genders = np.random.choice(GENDERS, rows)
    countries = np.random.choice(COUNTRIES, rows)

    account_age_days = np.random.randint(1, 1500, rows)
    last_login_days = np.random.randint(0, 60, rows)

    orders_count = np.random.poisson(lam=5, size=rows)
    support_tickets = np.random.poisson(lam=1, size=rows)

    membership = np.random.choice(MEMBERSHIP, rows, p=[0.4, 0.3, 0.2, 0.1])

    total_spent = np.round(np.random.exponential(scale=200, size=rows), 2)

    currency = [CURRENCY_MAP[c][0] for c in countries]
    usd_rate = [CURRENCY_MAP[c][1] for c in countries]
    total_spent_usd = np.round(total_spent * usd_rate, 2)

    churn_prob = (
        (last_login_days > 30).astype(int) * 0.4 +
        (orders_count < 2).astype(int) * 0.3 +
        (membership == "Free").astype(int) * 0.2 +
        (support_tickets > 3).astype(int) * 0.1
    )
    churn_prob = np.clip(churn_prob + np.random.uniform(-0.1, 0.1, rows), 0, 1)
    churn = (churn_prob > 0.5).astype(int)

    return pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "country": countries,
        "currency": currency,
        "account_age_days": account_age_days,
        "last_login_days": last_login_days,
        "orders_count": orders_count,
        "support_tickets": support_tickets,
        "membership": membership,
        "total_spent": total_spent,
        "total_spent_usd": total_spent_usd,
        "churn": churn
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--file", type=str, default="churn.csv")
    parser.add_argument("--mode", type=str, choices=["fresh", "append"], default="fresh")
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    filepath = f"data/raw/{args.file}"

    if args.mode == "fresh" or not os.path.exists(filepath):
        df = generate_dataset(args.rows)
        df.to_csv(filepath, index=False)
        print(f"✅ Created NEW dataset: {filepath} ({args.rows} rows)")

    else:  # append mode
        existing = pd.read_csv(filepath)
        start_index = len(existing) + 1
        df_new = generate_dataset(args.rows, start_index=start_index)
        final_df = pd.concat([existing, df_new], ignore_index=True)
        final_df.to_csv(filepath, index=False)
        print(f"✅ APPENDED {args.rows} rows → {filepath} (Total: {len(final_df)} rows)")
