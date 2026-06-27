"""
CreditPulse - Model Card Generator
Phase 6: Generates an SR 11-7 compliant model card in Markdown format.
"""

from pathlib import Path
from datetime import date

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


def generate_model_card():
    """Generate the full model card as a Markdown document."""

    card = f"""# CreditPulse Model Card
**Version:** 0.1.0
**Date:** {date.today().strftime("%B %d, %Y")}
**Framework:** SR 11-7 (Federal Reserve Model Risk Management Guidance)

---

## 1. Model Overview

| Field | Details |
|-------|---------|
| **Model Name** | CreditPulse Credit Risk Assessment System |
| **Version** | 0.1.0 |
| **Type** | Ensemble (XGBoost + Bayesian Bootstrap + Survival Forest + Isolation Forest) |
| **Purpose** | Predict probability of loan default for East African digital lending markets |
| **Primary Users** | Loan officers, portfolio managers, risk teams |
| **Regulatory Context** | IFRS 9, CBK Digital Credit Regulations 2022, SR 11-7 |

---

## 2. Training Data

| Field | Details |
|-------|---------|
| **Primary Dataset** | Home Credit Default Risk (Kaggle) |
| **Training Rows** | 307,511 loan applications |
| **Default Rate** | 8.07% (class imbalance addressed via scale_pos_weight=11) |
| **Features Used** | 12 features across credit bureau scores, income, employment, loan amounts |
| **Simulated Features** | M-Pesa mobile money features (calibrated to World Bank FinScope Kenya) |
| **Time Period** | Cross-sectional — no temporal split available in source data |

---

## 3. Model Performance

### XGBoost Classifier (Primary Scoring Model)
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| AUC-ROC | 0.7415 | > 0.70 | ✅ Pass |
| Gini Coefficient | 0.4830 | > 0.40 | ✅ Pass |
| Recall (Defaults) | 63% | > 50% | ✅ Pass |
| Precision (Defaults) | 17% | > 10% | ✅ Pass |

### Random Survival Forest (Time-to-Default)
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| C-index | 0.8130 | > 0.70 | ✅ Pass |

### Fraud Detector (Isolation Forest)
| Metric | Value |
|--------|-------|
| Flagged Applications | 5,379 (1.7%) |
| Default Rate — Normal | 8.1% |
| Default Rate — Flagged | 8.7% |

---

## 4. Causal Evidence

CreditPulse goes beyond correlation by providing causal evidence for credit decisions:

### Difference-in-Differences (COVID Natural Experiment)
- **DiD Coefficient:** -0.1125
- **P-value:** 0.002
- **Interpretation:** Income shocks causally affect default behaviour.
  Vulnerable occupation borrowers respond to shocks differently from
  stable occupation borrowers.

### Regression Discontinuity (Credit Score Cut-off)
- **LATE:** 0.0058
- **P-value:** 0.0028
- **95% CI:** [0.0020, 0.0095]
- **Interpretation:** Crossing the credit score cut-off has a real causal
  effect on default outcomes, not merely a predictive association.

---

## 5. Fairness Audit

### By Age Group
| Segment | Count | Approval Rate | TPR | Avg Risk Score |
|---------|-------|--------------|-----|----------------|
| Young (< 30) | 37,281 | 17.1% | 98.3% | 0.501 |
| Middle (30-45) | 103,271 | 30.2% | 95.6% | 0.428 |
| Senior (45+) | 70,568 | 43.8% | 91.2% | 0.356 |

**Finding:** Young borrowers face lower approval rates. This reflects genuine
higher risk (11.6% default rate vs 6.8% for seniors) rather than
discriminatory scoring. Recommend monitoring quarterly.

### By Occupation Type
| Segment | Count | Approval Rate | TPR | Default Rate |
|---------|-------|--------------|-----|-------------|
| Vulnerable occupation | 88,549 | 26.0% | 95.9% | 10.9% |
| Stable occupation | 122,571 | 37.1% | 94.3% | 7.3% |

**Finding:** TPR gap is < 2% — model treats both groups with near-equal
sensitivity. Approval rate difference reflects genuine risk differential.

---

## 6. Explainability

- **Method:** SHAP TreeExplainer (global + local explanations)
- **Top Features:** ext_source_mean, EXT_SOURCE_3, AMT_ANNUITY, AMT_CREDIT
- **Borrower Explanations:** Plain-language explanations generated for
  every decision via rule-based engine (Anthropic API-ready)
- **Proxy Discrimination:** No high-risk interaction terms identified
  between geographic proxies and default outcome

---

## 7. IFRS 9 Stage Assignment

| Stage | Criteria | Count (estimated) |
|-------|----------|-------------------|
| Stage 1 | pred_prob < 5% | ~180,000 |
| Stage 2 | 5% ≤ pred_prob < 20% | ~100,000 |
| Stage 3 | pred_prob ≥ 20% | ~27,000 |

---

## 8. Known Limitations

1. **No temporal validation** — model trained on cross-sectional data.
   Performance on future cohorts may differ.
2. **M-Pesa features are simulated** — real mobile money integration
   required for production deployment.
3. **Home Credit dataset** — sourced from Eastern Europe/Asia context.
   Requires recalibration on Kenyan loan data before production use.
4. **Class imbalance** — low precision (17%) means many false positives.
   Threshold should be tuned per lender's risk appetite.
5. **Young borrower gap** — approval rate disparity requires quarterly
   monitoring to ensure it reflects risk, not bias.

---

## 9. Monitoring Plan

| Metric | Frequency | Trigger for Retraining |
|--------|-----------|----------------------|
| AUC-ROC | Monthly | Drop below 0.70 |
| Default rate vs predicted | Monthly | PSI > 0.25 |
| Fairness metrics by segment | Quarterly | TPR gap > 5% |
| Fraud flag rate | Weekly | Spike > 3% |

---

## 10. Approval & Sign-off

| Role | Name | Date |
|------|------|------|
| Model Developer | B. Omare | {date.today().strftime("%B %d, %Y")} |
| Model Validator | Pending | — |
| Risk Officer | Pending | — |
| Compliance Officer | Pending | — |

---

*This model card was generated automatically by CreditPulse v0.1.0.
It follows the SR 11-7 model risk management framework and CBK
Digital Credit Regulations 2022.*
"""
    return card


def save_model_card(card):
    """Save model card as Markdown."""
    output = REPORTS / "model_card.md"
    with open(output, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"  Model card saved to {output}")
    print(f"  Length: {len(card.splitlines())} lines")


def main():
    print("Generating SR 11-7 Model Card...")
    card = generate_model_card()
    save_model_card(card)
    print("\nModel card complete!")


if __name__ == "__main__":
    main()