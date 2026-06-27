"""
CreditPulse - Regression Discontinuity
Phase 3: Estimates the causal effect of credit score cut-offs on default.
Borrowers just above and just below the cut-off are nearly identical,
so comparing their outcomes gives a causal estimate.
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


def prepare_rd_data(df, cutoff=0.5, bandwidth=0.15):
    """
    Prepare data for RD analysis.
    - Running variable: ext_source_mean (normalised credit score proxy)
    - Cut-off: 0.5 (above = lower risk, below = higher risk)
    - Bandwidth: only use borrowers within 0.15 of the cut-off
    """
    print(f"\nPreparing RD data (cut-off={cutoff}, bandwidth={bandwidth})...")

    df["score_centred"] = df["ext_source_mean"] - cutoff
    df["above_cutoff"] = (df["ext_source_mean"] >= cutoff).astype(int)

    df_rd = df[
        (df["ext_source_mean"] >= cutoff - bandwidth) &
        (df["ext_source_mean"] <= cutoff + bandwidth)
    ].copy()

    print(f"  Borrowers in bandwidth: {len(df_rd):,}")
    print(f"  Above cut-off: {df_rd['above_cutoff'].sum():,}")
    print(f"  Below cut-off: {(df_rd['above_cutoff'] == 0).sum():,}")

    return df_rd


def run_rd_regression(df_rd):
    """
    Fit separate regression lines on each side of the cut-off.
    The jump at the cut-off is the Local Average Treatment Effect (LATE).
    """
    print("\nRunning RD regression...")

    formula = "TARGET ~ score_centred * above_cutoff"
    model = smf.ols(formula, data=df_rd)
    result = model.fit()

    late = result.params["above_cutoff"]
    pval = result.pvalues["above_cutoff"]
    ci_lo = result.conf_int().loc["above_cutoff", 0]
    ci_hi = result.conf_int().loc["above_cutoff", 1]

    print("\n── RD Results ───────────────────────────────")
    print(f"  LATE (causal effect at cut-off): {late:.4f}")
    print(f"  P-value:                         {pval:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    sig = "SIGNIFICANT" if pval < 0.05 else "NOT significant"
    print(f"  Result: {sig} at 5% level")

    return result, late


def plot_rd(df_rd, cutoff=0.5):
    """Plot the RD discontinuity chart."""
    print("\nPlotting RD chart...")

    # Use fixed-width bins manually instead of pd.cut
    df_rd = df_rd.copy()
    df_rd["score_mid"] = (df_rd["ext_source_mean"] * 30).round() / 30

    binned = (
        df_rd.groupby("score_mid", observed=True)["TARGET"]
        .mean()
        .reset_index()
    )

    # Ensure float type
    binned["score_mid"] = binned["score_mid"].astype(float)
    binned["TARGET"] = binned["TARGET"].astype(float)

    below = binned[binned["score_mid"] < cutoff].copy()
    above = binned[binned["score_mid"] >= cutoff].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(below["score_mid"], below["TARGET"],
               color="#E74C3C", alpha=0.7, label="Below cut-off (higher risk)")
    ax.scatter(above["score_mid"], above["TARGET"],
               color="#3498DB", alpha=0.7, label="Above cut-off (lower risk)")

    # Fit trend lines
    for subset, colour in [(below, "#E74C3C"), (above, "#3498DB")]:
        if len(subset) > 1:
            x = subset["score_mid"].values
            y = subset["TARGET"].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color=colour, linewidth=2.5)

    ax.axvline(x=cutoff, color="black", linestyle="--",
               linewidth=1.5, label=f"Cut-off = {cutoff}")

    ax.set_title("Regression Discontinuity: Credit Score Cut-off Effect on Default",
                 fontweight="bold")
    ax.set_xlabel("Credit Score Proxy (ext_source_mean)")
    ax.set_ylabel("Default Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    output = REPORTS / "rd_plot.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  Plot saved to {output}")


def main():
    df = load_data()
    df_rd = prepare_rd_data(df)
    result, late = run_rd_regression(df_rd)
    plot_rd(df_rd)
    print("\nRegression Discontinuity analysis complete!")


if __name__ == "__main__":
    main()