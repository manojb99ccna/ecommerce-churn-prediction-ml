import pandas as pd
import numpy as np
import random
import os

# Define paths
RAW_DATA_PATH = "data/raw"

def generate_dataset(file_name="churn_full.csv", rows=10000, mode="fresh"):
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    if mode == "append" and os.path.exists(os.path.join(RAW_DATA_PATH, file_name)):
        df_existing = pd.read_csv(os.path.join(RAW_DATA_PATH, file_name))
    else:
        df_existing = pd.DataFrame()

    np.random.seed(42)
    random.seed(42)

    countries = ["USA", "Canada", "UK", "Germany", "India", "Australia", "Bangladesh", "Sri Lanka", "Pakistan"]
    currencies = ["USD", "CAD", "GBP", "EUR", "INR", "AUD", "BDT", "LKR", "PKR"]
    memberships = ["Silver", "Gold", "Platinum"]
    genders = ["Male", "Female", "Other"]

    data = []
    for i in range(rows):
        customer_id = f"CUST{10000 + i}"
        age = np.random.randint(18, 70)
        account_age_days = np.random.randint(30, 2000)
        last_login_days = np.random.randint(0, 60)
        total_spent = np.random.uniform(10, 2000)
        orders_count = np.random.randint(1, 50)
        support_tickets = np.random.randint(0, 6)
        gender = random.choice(genders)
        country = random.choice(countries)
        membership = random.choice(memberships)
        currency = currencies[countries.index(country)]

        # === Intelligent churn logic ===
        churn_probability = 0.1  # Base

        # Inactivity → churn
        if last_login_days > 30:
            churn_probability += 0.4

        # Few orders or low spending → churn
        if orders_count < 3 or total_spent < 100:
            churn_probability += 0.3

        # Too many support tickets → churn
        if support_tickets > 3:
            churn_probability += 0.2

        # Membership effect
        if membership == "Silver":
            churn_probability += 0.1
        elif membership == "Platinum":
            churn_probability -= 0.1

        # Cap probability between 0–1
        churn_probability = np.clip(churn_probability, 0, 1)
        churned = np.random.rand() < churn_probability

        data.append([
            customer_id, age, account_age_days, last_login_days,
            total_spent, orders_count, support_tickets, gender,
            country, membership, currency, churned
        ])

    df_new = pd.DataFrame(data, columns=[
        "customer_id", "age", "account_age_days", "last_login_days",
        "total_spent", "orders_count", "support_tickets", "gender",
        "country", "membership", "currency", "churned"
    ])

    df_final = pd.concat([df_existing, df_new], ignore_index=True)
    df_final.to_csv(os.path.join(RAW_DATA_PATH, file_name), index=False)
    print(f"✅ Saved {len(df_final)} records → {os.path.join(RAW_DATA_PATH, file_name)}")

if __name__ == "__main__":
    generate_dataset(file_name="churn_full.csv", rows=10000, mode="fresh")
