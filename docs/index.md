# CreditPulse Documentation

**Causal AI System for Equitable Credit Risk Assessment**

## Overview

CreditPulse is a full-stack credit risk intelligence platform built for East African digital lending markets. It combines causal reasoning, Bayesian uncertainty quantification, NLP analysis of M-Pesa transactions, and explainable AI.

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/B-Omare/creditpulse.git
cd creditpulse
conda create -n creditpulse python=3.11 -y
conda activate creditpulse
pip install -e ".[dev]"

# 2. Download data
pip install kaggle
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/raw/

# 3. Run the pipeline
snakemake --cores 4

# 4. Start the API
uvicorn creditpulse.api.main:app --reload

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

## Architecture

```
Raw Data (Kaggle)
    ↓
Ingestion & ETL (creditpulse/ingestion/)
    ↓
Feature Engineering → IFRS 9 Schema
    ↓
┌─────────────────────────────────┐
│  Causal Analysis (DoWhy/DiD)    │
│  ML Models (XGBoost/Bayesian)   │
│  NLP (BERT/BERTopic on M-Pesa)  │
└─────────────────────────────────┘
    ↓
FastAPI (creditpulse/api/)
    ↓
Streamlit Dashboard (app/)
```

## Modules

- **ingestion/** — ETL, feature engineering, IFRS 9 schema, M-Pesa simulation
- **causal/** — DoWhy causal graphs, Difference-in-Differences, Regression Discontinuity
- **nlp/** — M-Pesa NLP, BERTopic clustering, RAG chatbot for loan officers
- **models/** — XGBoost classifier, Bayesian credit scoring, Survival Forest
- **explainability/** — SHAP values, LIME, SR 11-7 model card
- **api/** — FastAPI REST endpoints

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/predict` | POST | Credit risk assessment |
| `/explain/{id}` | GET | SHAP explanation for a borrower |

## Running Tests

```bash
pytest tests/ -v --cov=creditpulse
```
