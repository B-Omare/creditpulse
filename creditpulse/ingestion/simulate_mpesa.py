"""
Simulate realistic M-Pesa transaction sequences for East African context.
Generates synthetic mobile money data with behavioral patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/simulated/mpesa_transactions.parquet")

MPESA_CATEGORIES = [
    "SEND_MONEY", "RECEIVE_MONEY", "BUY_AIRTIME", "PAY_BILL",
    "WITHDRAW_AGENT", "DEPOSIT_AGENT", "LIPA_NA_MPESA",
    "FULIZA_REPAY", "FULIZA_BORROW", "SALARY_PAYMENT",
    "SCHOOL_FEES", "RENT_PAYMENT", "GROCERY_PAYMENT",
]

MERCHANTS = [
    "Naivas Supermarket", "Nakumatt", "Java House", "KFC Kenya",
    "Equity Bank Agent", "KCB Agent", "M-Kopa Solar",
    "KPLC Token", "Nairobi Water", "Safaricom Airtime",
    "Mama Mboga - Gikomba", "Hardware Shop - Mombasa Rd",
    "Farm Input - Eldoret", "Maize Mill - Nakuru",
]

NOTE_TEMPLATES = [
    "School fees for {month}",
    "House rent {month}",
    "Mpesa to {name}",
    "From {name} - business",
    "Salary - {employer}",
    "Loan repayment week {week}",
    "Emergency - hospital",
    "Farming supplies",
    "Airtime recharge",
    "Pay bill - electricity",
]

EMPLOYERS = ["Safaricom Ltd", "Kenya Airways", "Equity Bank", "County Govt",
             "School Teacher", "NGO Worker", "Self Employed", "Tea Factory"]

NAMES = ["John Kamau", "Mary Wanjiku", "Peter Omondi", "Grace Akinyi",
         "James Mwangi", "Faith Njeri", "David Ouma", "Susan Waweru"]


def generate_borrower_transactions(
    borrower_id: int,
    n_months: int = 12,
    income_level: str = "medium",
    is_defaulter: bool = False,
) -> pd.DataFrame:
    income_map = {"low": (3000, 8000), "medium": (15000, 45000), "high": (60000, 200000)}
    income_range = income_map.get(income_level, income_map["medium"])

    records = []
    start_date = datetime(2023, 1, 1)

    monthly_income = np.random.uniform(*income_range)

    for month in range(n_months):
        month_date = start_date + timedelta(days=30 * month)
        n_transactions = np.random.poisson(25 if not is_defaulter else 15)

        # Salary inflow (most borrowers)
        if np.random.random() < 0.8:
            records.append({
                "borrower_id": borrower_id,
                "transaction_date": month_date + timedelta(days=np.random.randint(1, 5)),
                "amount": monthly_income * np.random.uniform(0.9, 1.1),
                "transaction_type": "RECEIVE_MONEY",
                "category": "SALARY_PAYMENT",
                "description": f"Salary - {np.random.choice(EMPLOYERS)}",
                "balance_after": monthly_income * np.random.uniform(1.0, 1.5),
                "is_defaulter": is_defaulter,
            })

        for _ in range(n_transactions):
            txn_date = month_date + timedelta(days=np.random.randint(0, 30))
            category = np.random.choice(MPESA_CATEGORIES)
            amount = abs(np.random.lognormal(mean=8.5, sigma=1.2))  # KES

            template = np.random.choice(NOTE_TEMPLATES)
            description = template.format(
                month=txn_date.strftime("%B"),
                name=np.random.choice(NAMES),
                employer=np.random.choice(EMPLOYERS),
                week=np.random.randint(1, 4),
            )

            records.append({
                "borrower_id": borrower_id,
                "transaction_date": txn_date,
                "amount": round(amount, 2),
                "transaction_type": "SEND_MONEY" if "PAY" in category or "SEND" in category else "RECEIVE_MONEY",
                "category": category,
                "description": description,
                "balance_after": round(abs(np.random.normal(5000, 3000)), 2),
                "is_defaulter": is_defaulter,
            })

    return pd.DataFrame(records)


def simulate(
    n_borrowers: int = 1000,
    output_path: Path = OUTPUT_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    all_dfs = []
    income_levels = ["low", "medium", "high"]
    default_rate = 0.12  # 12% default rate (realistic for East Africa)

    logger.info(f"Simulating M-Pesa transactions for {n_borrowers} borrowers...")

    for i in range(n_borrowers):
        income_level = np.random.choice(income_levels, p=[0.4, 0.45, 0.15])
        is_defaulter = np.random.random() < default_rate
        df = generate_borrower_transactions(
            borrower_id=i,
            income_level=income_level,
            is_defaulter=is_defaulter,
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["transaction_date"] = pd.to_datetime(combined["transaction_date"])
    combined = combined.sort_values(["borrower_id", "transaction_date"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    logger.info(f"Simulated {len(combined):,} M-Pesa transactions → {output_path}")
    return combined


if __name__ == "__main__":
    simulate()
