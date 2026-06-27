"""
Phase 4 — NLP analysis of M-Pesa transaction descriptions.
Extracts behavioral signals using BERT embeddings and BERTopic clustering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/simulated/mpesa_transactions.parquet")


STRESS_KEYWORDS = [
    "emergency", "hospital", "urgent", "borrow", "fuliza",
    "overdue", "late", "penalty", "debt", "loan repay",
]

STABILITY_KEYWORDS = [
    "salary", "business", "school fees", "rent", "savings",
    "investment", "farm", "harvest",
]


def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based NLP features from transaction descriptions."""
    desc = df["description"].fillna("").str.lower()

    df["stress_signal"] = desc.apply(
        lambda x: sum(1 for kw in STRESS_KEYWORDS if kw in x)
    )
    df["stability_signal"] = desc.apply(
        lambda x: sum(1 for kw in STABILITY_KEYWORDS if kw in x)
    )
    df["has_emergency"] = desc.str.contains("emergency|hospital|urgent", regex=True).astype(int)
    df["is_salary"] = desc.str.contains("salary|pay slip|employer", regex=True).astype(int)
    df["is_loan_related"] = desc.str.contains("loan|fuliza|borrow|repay", regex=True).astype(int)

    return df


def aggregate_borrower_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level NLP signals to borrower level."""
    features = df.groupby("borrower_id").agg(
        total_transactions=("amount", "count"),
        total_outflow=("amount", lambda x: x[df.loc[x.index, "transaction_type"] == "SEND_MONEY"].sum()),
        total_inflow=("amount", lambda x: x[df.loc[x.index, "transaction_type"] == "RECEIVE_MONEY"].sum()),
        avg_stress_signal=("stress_signal", "mean"),
        max_stress_signal=("stress_signal", "max"),
        avg_stability_signal=("stability_signal", "mean"),
        n_emergency_txns=("has_emergency", "sum"),
        n_salary_txns=("is_salary", "sum"),
        n_loan_txns=("is_loan_related", "sum"),
        avg_balance=("balance_after", "mean"),
        min_balance=("balance_after", "min"),
        balance_volatility=("balance_after", "std"),
    ).reset_index()

    features["net_flow"] = features["total_inflow"] - features["total_outflow"]
    features["stress_score"] = (
        features["avg_stress_signal"] * 0.4
        + features["n_emergency_txns"] / (features["total_transactions"] + 1) * 0.3
        + features["n_loan_txns"] / (features["total_transactions"] + 1) * 0.3
    )

    return features


def run_bertopic(df: pd.DataFrame, n_topics: int = 10) -> dict:
    """Cluster transaction descriptions using BERTopic."""
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        docs = df["description"].dropna().sample(min(5000, len(df)), random_state=42).tolist()
        model = BERTopic(nr_topics=n_topics, verbose=False)
        topics, probs = model.fit_transform(docs)
        topic_info = model.get_topic_info()
        logger.info(f"BERTopic found {len(topic_info)} topics")
        return {"topics": topic_info.to_dict(), "n_docs": len(docs)}

    except ImportError:
        logger.warning("BERTopic not installed")
        return {"error": "bertopic not installed"}


def analyse(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df = extract_text_features(df)
    features = aggregate_borrower_signals(df)
    logger.info(f"NLP features extracted for {len(features)} borrowers")
    return features


if __name__ == "__main__":
    analyse()
