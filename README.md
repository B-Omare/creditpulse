# CreditPulse 💳

**A Causal AI System for Equitable Credit Risk Assessment**

[![CI](https://github.com/B-Omare/creditpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/B-Omare/creditpulse/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *A full-stack credit risk intelligence platform built for East African digital lending markets.*

CreditPulse goes beyond a simple credit score by combining:

- 🔍 **Causal reasoning** — understanding *WHY* borrowers default, not just who does
- 📊 **Bayesian uncertainty** — a probability range, not a single score
- 📱 **M-Pesa NLP** — reading transaction descriptions for behavioural signals  
- 🧾 **Explainable AI** — plain-English explanations for borrowers and regulators

Built for the East African context: M-Pesa mobile money, thin credit files, smallholder farmers, and CBK regulatory standards.

---

## Quickstart

```bash
git clone https://github.com/B-Omare/creditpulse.git
cd creditpulse

conda create -n creditpulse python=3.11 -y
conda activate creditpulse
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start the API
uvicorn creditpulse.api.main:app --reload

# Launch the Streamlit dashboard
streamlit run app/streamlit_app.py
```

## Architecture

```
Raw Data (Kaggle / M-Pesa)
        │
        ▼
  Ingestion & ETL ──── IFRS 9 Schema
        │
        ├── Causal Analysis (DoWhy, DiD, RD)
        ├── ML Models (XGBoost, Bayesian, Survival)
        └── NLP (BERT, BERTopic, RAG Chatbot)
                │
                ▼
          FastAPI REST
                │
                ▼
      Streamlit Dashboard
     ┌──────┬──────┬──────┐
     │ Officer│Borrower│Regulator│
```

## The 7 Phases

| Phase | What You Build |
|-------|---------------|
| 1 | Project scaffold, Docker, CI/CD |
| 2 | Data ingestion, ETL, IFRS 9 schema |
| 3 | Causal inference (DoWhy, DiD, RD) |
| 4 | NLP — M-Pesa analysis + RAG chatbot |
| 5 | ML models (XGBoost, Bayesian, Survival Forest) |
| 6 | Explainability — SHAP, LIME, model card |
| 7 | FastAPI + Streamlit dashboard |

## Data Sources

| Dataset | Source |
|---------|--------|
| Home Credit Default Risk | [Kaggle](https://kaggle.com/competitions/home-credit-default-risk) |
| Lending Club | [Kaggle](https://kaggle.com/datasets/wordsforthewise/lending-club) |
| World Bank FinScope Kenya | [finmark.org.za](https://finmark.org.za/finscopekenya) |
| CBK Annual Reports | [centralbank.go.ke](https://centralbank.go.ke/publications) |
| M-Pesa (simulated) | Generated in Phase 2 |

## API

```bash
# Health check
curl http://localhost:8000/health

# Score a borrower
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_years": 32,
    "employment_years": 4,
    "amt_income_total": 240000,
    "amt_credit": 120000,
    "amt_annuity": 8000,
    "ext_source_mean": 0.62,
    "cnt_children": 1
  }'
```

Response:
```json
{
  "pd_point_estimate": 0.087,
  "pd_lower_95": 0.012,
  "pd_upper_95": 0.163,
  "ifrs9_stage": 1,
  "ecl_estimate": 4698.0,
  "recommendation": "APPROVE",
  "explanation": "PD of 8.7% (95% CI: 1.2%–16.3%). IFRS 9 Stage 1.",
  "risk_factors": []
}
```

## Docker

```bash
docker compose up
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## Pipeline

```bash
snakemake --cores 4
```

## License

MIT — see [LICENSE](LICENSE)

---

*Built by [B-Omare](https://github.com/B-Omare)*
