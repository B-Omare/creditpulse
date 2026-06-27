"""
CreditPulse - Difference-in-Differences
Phase 3: Estimates the causal effect of income shocks (COVID proxy)
on credit default using a natural experiment design.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


def load_data():
    """Load cleaned loan data."""
    print("Loading cleaned data...")
    df = pd.read_parquet(PROCESSED / "loans_clean.parquet")
    print(f"  Loaded {len(df):,} rows")
    return df


def create_experiment_variables(df):
    """
    Create DiD variables using available data as proxies.

    - post_shock: borrowers with high income volatility proxy
      (low EXT_SOURCE scores = more vulnerable to shocks)
    - treated: borrowers in high-risk occupations
      (Laborers, Drivers — most affected by COVID-type shocks)
    """
    print("Creating experiment variables...")

    # POST SHOCK: bottom 40% of credit bureau scores = shock-affected
    threshold = df["ext_source_mean"].quantile(0.40)
    df["post_shock"] = (df["ext_source_mean"] < threshold).astype(int)

    # TREATED: occupation types most vulnerable to income shocks
    vulnerable_occupations = [
        "Laborers", "Drivers", "Low-skill Laborers",
        "Cooking staff", "Security staff", "Waiters/barmen staff"
    ]
    df["treated"] = df["OCCUPATION_TYPE"].isin(vulnerable_occupations).astype(int)

    # Interaction term: the DiD estimator
    df["did"] = df["post_shock"] * df["treated"]

    print(f"  Post-shock group: {df['post_shock'].sum():,} borrowers")
    print(f"  Treated group:    {df['treated'].sum():,} borrowers")
    print(f"  Both (DiD cell):  {df['did'].sum():,} borrowers")

    return df


def run_did_regression(df):
    """
    Run the DiD regression.
    TARGET ~ post_shock + treated + post_shock*treated + controls
    The interaction term (did) is the causal estimate.
    """
    print("\nRunning DiD regression...")

    formula = ("TARGET ~ post_shock + treated + did "
               "+ income_to_credit_ratio + days_employed_pct "
               "+ AMT_CREDIT + ext_source_std")

    model = smf.logit(formula, data=df.dropna(subset=[
        "TARGET", "post_shock", "treated", "did",
        "income_to_credit_ratio", "days_employed_pct",
        "AMT_CREDIT", "ext_source_std"
    ]))

    result = model.fit(disp=False)

    print("\n── DiD Results ──────────────────────────────")
    print(f"  DiD coefficient (causal effect): {result.params['did']:.4f}")
    print(f"  P-value:                         {result.pvalues['did']:.4f}")
    print(f"  95% CI: [{result.conf_int().loc['did', 0]:.4f}, "
          f"{result.conf_int().loc['did', 1]:.4f}]")

    sig = "SIGNIFICANT" if result.pvalues["did"] < 0.05 else "NOT significant"
    print(f"  Result: {sig} at 5% level")

    return result


def plot_did(df):
    """Visualise the DiD parallel trends."""
    print("\nPlotting DiD chart...")

    summary = df.groupby(["treated", "post_shock"])["TARGET"].mean().reset_index()
    summary.columns = ["treated", "post_shock", "default_rate"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for group, label, colour in [(0, "Control (stable occupations)", "#3498DB"),
                                  (1, "Treated (vulnerable occupations)", "#E74C3C")]:
        d = summary[summary["treated"] == group].sort_values("post_shock")
        ax.plot(["Pre-shock", "Post-shock"], d["default_rate"].values,
                marker="o", linewidth=2.5, label=label, color=colour)

    ax.set_title("Difference-in-Differences: Effect of Income Shock on Default Rate",
                 fontweight="bold")
    ax.set_ylabel("Default Rate")
    ax.set_xlabel("Period")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    output = REPORTS / "did_plot.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  Plot saved to {output}")


def main():
    df = load_data()
    df = create_experiment_variables(df)
    result = run_did_regression(df)
    plot_did(df)
    print("\nDifference-in-Differences analysis complete!")


if __name__ == "__main__":
    main()