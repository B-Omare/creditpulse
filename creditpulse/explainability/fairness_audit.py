"""
CreditPulse - Fairness Audit
Phase 6: Measures model fairness across borrower segments.
Checks equal opportunity, demographic parity, and calibration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_score, recall_score
)

PROCESSED = Path("data/processed")
MODELS    = Path("models")
REPORTS   = Path("reports")


def load_data_and_model():
    """Load cleaned data and trained XGBoost model."""
    print("Loading data and model...")
    df    = pd.read_parquet(PROCESSED / "loans_clean.parquet")
    model = joblib.load(MODELS / "xgboost_model.pkl")

    FEATURES = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "ext_source_mean", "ext_source_std",
        "income_to_credit_ratio", "days_employed_pct",
        "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED",
    ]

    df = df[FEATURES + ["TARGET", "NAME_INCOME_TYPE",
                        "OCCUPATION_TYPE", "NAME_FAMILY_STATUS"]].dropna()
    print(f"  Loaded {len(df):,} rows")
    return df, model, FEATURES


def create_segments(df):
    """Define borrower segments for fairness analysis."""
    print("\nCreating borrower segments...")

    # Segment 1: Income type
    df["segment_income"] = df["NAME_INCOME_TYPE"].apply(
        lambda x: "Salaried" if x in ["Working", "Commercial associate",
                                       "State servant"]
        else "Self-employed/Other"
    )

    # Segment 2: Age group (using DAYS_BIRTH — negative values)
    age_years = np.abs(df["DAYS_BIRTH"]) / 365
    df["segment_age"] = pd.cut(
        age_years,
        bins=[0, 30, 45, 100],
        labels=["Young (< 30)", "Middle (30-45)", "Senior (45+)"]
    )

    # Segment 3: Occupation risk
    vulnerable = ["Laborers", "Drivers", "Low-skill Laborers",
                  "Security staff", "Cooking staff"]
    df["segment_occupation"] = df["OCCUPATION_TYPE"].apply(
        lambda x: "Vulnerable occupation" if x in vulnerable
        else "Stable occupation"
    )

    print(f"  Income segments: {df['segment_income'].value_counts().to_dict()}")
    return df


def compute_fairness_metrics(df, model, features, segment_col, segment_name):
    """Compute fairness metrics for a given segment."""
    print(f"\n── Fairness by {segment_name} ─────────────────────")

    results = []
    X = df[features]
    df = df.copy()
    df["pred_proba"] = model.predict_proba(X)[:, 1]
    df["pred_label"] = (df["pred_proba"] > 0.3).astype(int)

    for segment in df[segment_col].dropna().unique():
        mask = df[segment_col] == segment
        subset = df[mask]

        if len(subset) < 100:
            continue

        tn, fp, fn, tp = confusion_matrix(
            subset["TARGET"], subset["pred_label"],
            labels=[0, 1]
        ).ravel() if subset["TARGET"].nunique() == 2 else (0, 0, 0, 0)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Sensitivity
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        approval_rate = (subset["pred_label"] == 0).mean()  # Approved = not flagged
        avg_score = subset["pred_proba"].mean()

        results.append({
            "Segment": str(segment),
            "Count": len(subset),
            "Default Rate": f"{subset['TARGET'].mean():.1%}",
            "Approval Rate": f"{approval_rate:.1%}",
            "TPR (Recall)": f"{tpr:.1%}",
            "FPR": f"{fpr:.1%}",
            "Precision": f"{precision:.1%}",
            "Avg Risk Score": f"{avg_score:.3f}",
        })

        print(f"  {str(segment)[:30]:<30} | "
              f"n={len(subset):>6,} | "
              f"Approval={approval_rate:.1%} | "
              f"TPR={tpr:.1%} | "
              f"Precision={precision:.1%}")

    return pd.DataFrame(results)


def plot_fairness(df, model, features):
    """Plot fairness comparison across segments."""
    print("\nPlotting fairness charts...")

    X = df[features]
    df = df.copy()
    df["pred_proba"] = model.predict_proba(X)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 1: Average risk score by income type
    ax = axes[0]
    income_scores = df.groupby("segment_income")["pred_proba"].mean()
    bars = ax.bar(income_scores.index, income_scores.values,
                  color=["#3498DB", "#E74C3C"], alpha=0.8)
    ax.set_title("Avg Risk Score\nby Income Type", fontweight="bold")
    ax.set_ylabel("Average Predicted Default Probability")
    for bar, val in zip(bars, income_scores.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Chart 2: Average risk score by age group
    ax = axes[1]
    age_scores = df.groupby("segment_age", observed=True)["pred_proba"].mean()
    bars = ax.bar(age_scores.index.astype(str), age_scores.values,
                  color=["#E67E22", "#3498DB", "#2ECC71"], alpha=0.8)
    ax.set_title("Avg Risk Score\nby Age Group", fontweight="bold")
    ax.set_ylabel("Average Predicted Default Probability")
    for bar, val in zip(bars, age_scores.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Chart 3: Average risk score by occupation
    ax = axes[2]
    occ_scores = df.groupby("segment_occupation")["pred_proba"].mean()
    bars = ax.bar(occ_scores.index, occ_scores.values,
                  color=["#E74C3C", "#2ECC71"], alpha=0.8)
    ax.set_title("Avg Risk Score\nby Occupation Type", fontweight="bold")
    ax.set_ylabel("Average Predicted Default Probability")
    for bar, val in zip(bars, occ_scores.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("CreditPulse — Fairness Audit Across Borrower Segments",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    output = REPORTS / "fairness_audit.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fairness chart saved to {output}")


def save_fairness_report(results_list):
    """Save all fairness metrics to CSV."""
    all_results = pd.concat(results_list, ignore_index=True)
    output = REPORTS / "fairness_metrics.csv"
    all_results.to_csv(output, index=False)
    print(f"\n  Fairness metrics saved to {output}")
    return all_results


def main():
    df, model, features = load_data_and_model()
    df = create_segments(df)

    results = []
    results.append(compute_fairness_metrics(
        df, model, features, "segment_income", "Income Type"))
    results.append(compute_fairness_metrics(
        df, model, features, "segment_age", "Age Group"))
    results.append(compute_fairness_metrics(
        df, model, features, "segment_occupation", "Occupation"))

    plot_fairness(df, model, features)
    save_fairness_report(results)
    print("\nFairness audit complete!")


if __name__ == "__main__":
    main()