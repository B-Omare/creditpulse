"""
CreditPulse - BERTopic Topic Modelling
Phase 4: Discovers latent themes in loan officer notes using BERTopic.
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic
from pathlib import Path

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


def simulate_loan_notes(n=500, random_state=42):
    """
    Simulate realistic loan officer notes for East African context.
    In production these would be real notes from the loan management system.
    """
    print(f"Simulating {n} loan officer notes...")
    rng = np.random.default_rng(random_state)

    templates = [
        # Repayment stress
        "Borrower reported reduced income due to drought affecting farm yields. Requested loan restructuring.",
        "Client works as matatu driver. Income irregular due to fuel price increases. Missed last two payments.",
        "Small business owner in Gikomba market. Sales down 40% since inflation spike. At risk of default.",
        "Borrower has three concurrent mobile loans. Showing signs of loan stacking. High risk flag raised.",
        "Client M-Pesa balance declining for 30 days. Income volatility noted. Early warning triggered.",

        # Positive signals
        "Salaried employee at county government. Consistent M-Pesa salary deposits. Low risk profile.",
        "Borrower operates successful posho mill. Regular cash flows observed in transaction history.",
        "Client has repaid four previous loans on time. Credit bureau score improving. Recommend limit increase.",
        "Teacher with TSC payslip. Stable employment. M-Pesa balance healthy. Approved for top-up loan.",
        "Borrower runs mobile money agent business. High transaction volume. Strong repayment capacity.",

        # Fraud signals
        "Unusual transaction pattern detected. Multiple accounts sending funds before loan application.",
        "Borrower provided employer details that could not be verified. Documents appear altered.",
        "Application submitted from same device as three previous defaulted accounts. Fraud risk high.",
        "Inconsistent income declarations across multiple loan applications. Refer to fraud team.",

        # Restructuring
        "Borrower affected by El Nino floods in Kisumu. Crop loss confirmed. Restructuring approved.",
        "Client lost employment. Severance pay received. Agreed repayment holiday for 60 days.",
        "Medical emergency reported. Hospital receipts verified. Loan restructured over 12 months.",
        "Borrower relocated to Mombasa for work. New income source confirmed. Repayment plan updated.",
    ]

    notes = []
    for _ in range(n):
        base = templates[rng.integers(0, len(templates))]
        notes.append(base)

    print(f"  Generated {len(notes)} notes")
    return notes


def run_topic_model(notes):
    """Fit BERTopic and extract themes from loan notes."""
    print("\nFitting BERTopic model...")
    print("  This may take 2-3 minutes...")

    topic_model = BERTopic(
        language="english",
        calculate_probabilities=True,
        verbose=False,
        min_topic_size=10,
    )

    topics, probs = topic_model.fit_transform(notes)

    print("\n── Discovered Topics ─────────────────────────")
    topic_info = topic_model.get_topic_info()
    for _, row in topic_info.iterrows():
        if row["Topic"] != -1:  # -1 is the outlier topic
            print(f"  Topic {row['Topic']}: {row['Name']}")
            print(f"    Count: {row['Count']} notes")
            print()

    return topic_model, topics


def save_topic_summary(topic_model):
    """Save topic summary to reports folder."""
    topic_info = topic_model.get_topic_info()
    output = REPORTS / "bertopic_summary.csv"
    topic_info.to_csv(output, index=False)
    print(f"  Topic summary saved to {output}")


def main():
    notes = simulate_loan_notes(n=500)
    topic_model, topics = run_topic_model(notes)
    save_topic_summary(topic_model)
    print("\nBERTopic analysis complete!")


if __name__ == "__main__":
    main()