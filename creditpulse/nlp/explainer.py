"""
CreditPulse - Loan Decision Explainer
Phase 4: Generates plain-language explanations for loan decisions.
Uses a rule-based engine locally. Swap anthropic_explain() for
production when an API key is available.
"""

from pathlib import Path

# ── Rule-based explainer (free, no API needed) ─────────────────────────────

def rule_based_explain(borrower_data: dict, decision: str) -> str:
    """
    Generate a plain-language explanation based on borrower features.
    This runs entirely locally with no API cost.
    """
    name     = borrower_data.get("name", "the borrower")
    score    = borrower_data.get("ext_source_mean", 0.5)
    ratio    = borrower_data.get("income_to_credit_ratio", 0.5)
    employed = borrower_data.get("days_employed_pct", 0.5)
    balance  = borrower_data.get("mpesa_avg_balance_30d", 5000)

    # ── Identify the strongest risk factors ───────────────────────────────
    risk_factors = []
    strengths    = []

    if score < 0.4:
        risk_factors.append("a low credit bureau score")
    elif score > 0.6:
        strengths.append("a strong credit bureau score")

    if ratio < 0.3:
        risk_factors.append("a high loan amount relative to income")
    elif ratio > 0.6:
        strengths.append("a healthy income-to-loan ratio")

    if employed < 0.1:
        risk_factors.append("a short employment history")
    elif employed > 0.3:
        strengths.append("a stable employment history")

    if balance < 2000:
        risk_factors.append("a low average M-Pesa balance")
    elif balance > 8000:
        strengths.append("a healthy M-Pesa balance")

    # ── Build the explanation ─────────────────────────────────────────────
    if decision.lower() == "approved":
        if strengths:
            strength_text = ", ".join(strengths)
            explanation = (
                f"Your loan has been approved. "
                f"Our assessment found {strength_text}, "
                f"which indicates a reliable repayment capacity. "
                f"Please ensure timely repayments to maintain your "
                f"credit profile and qualify for larger loans in future."
            )
        else:
            explanation = (
                f"Your loan has been approved based on your overall profile. "
                f"Please ensure timely repayments to strengthen your "
                f"credit record for future applications."
            )
    else:  # declined
        if risk_factors:
            risk_text = ", ".join(risk_factors)
            explanation = (
                f"Unfortunately your loan application was not approved at this time. "
                f"The main factors were {risk_text}. "
                f"To improve your chances: build your M-Pesa savings over 3 months, "
                f"reduce any existing loan balances, and apply for a smaller amount "
                f"that better matches your current income level."
            )
        else:
            explanation = (
                f"Unfortunately your loan application was not approved at this time. "
                f"We encourage you to maintain consistent M-Pesa activity and "
                f"reapply in 3 months."
            )

    return explanation


def explain_decision(borrower_data: dict, decision: str) -> str:
    """
    Main entry point for generating loan explanations.
    Returns a plain-language explanation for the borrower.
    """
    explanation = rule_based_explain(borrower_data, decision)
    return explanation


def main():
    """Test the explainer with sample borrowers."""

    test_cases = [
        {
            "borrower": {
                "name": "Jane Wanjiku",
                "ext_source_mean": 0.65,
                "income_to_credit_ratio": 0.70,
                "days_employed_pct": 0.35,
                "mpesa_avg_balance_30d": 12000,
            },
            "decision": "approved"
        },
        {
            "borrower": {
                "name": "John Kamau",
                "ext_source_mean": 0.30,
                "income_to_credit_ratio": 0.20,
                "days_employed_pct": 0.05,
                "mpesa_avg_balance_30d": 800,
            },
            "decision": "declined"
        },
        {
            "borrower": {
                "name": "Mary Achieng",
                "ext_source_mean": 0.50,
                "income_to_credit_ratio": 0.45,
                "days_employed_pct": 0.25,
                "mpesa_avg_balance_30d": 4500,
            },
            "decision": "declined"
        },
    ]

    print("── CreditPulse Loan Decision Explainer ──────────────────\n")
    for case in test_cases:
        borrower  = case["borrower"]
        decision  = case["decision"]
        explanation = explain_decision(borrower, decision)

        print(f"Borrower:   {borrower['name']}")
        print(f"Decision:   {decision.upper()}")
        print(f"Explanation: {explanation}")
        print()


if __name__ == "__main__":
    main()