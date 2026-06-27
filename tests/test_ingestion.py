"""Tests for data ingestion and feature engineering."""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from pathlib import Path

from creditpulse.ingestion.clean import fix_anomalies, impute_missing, encode_categoricals
from creditpulse.ingestion.features import build_features
from creditpulse.ingestion.ifrs9 import assign_stage, compute_ecl
from creditpulse.ingestion.simulate_mpesa import generate_borrower_transactions


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "SK_ID_CURR": range(n),
        "TARGET": np.random.binomial(1, 0.1, n),
        "AMT_INCOME_TOTAL": np.random.normal(150000, 50000, n),
        "AMT_CREDIT": np.random.normal(500000, 200000, n),
        "AMT_ANNUITY": np.random.normal(25000, 5000, n),
        "AMT_GOODS_PRICE": np.random.normal(450000, 180000, n),
        "DAYS_BIRTH": np.random.randint(-25000, -8000, n),
        "DAYS_EMPLOYED": np.random.randint(-5000, -100, n),
        "EXT_SOURCE_1": np.random.uniform(0, 1, n),
        "EXT_SOURCE_2": np.random.uniform(0, 1, n),
        "EXT_SOURCE_3": np.random.uniform(0, 1, n),
        "CNT_CHILDREN": np.random.randint(0, 5, n),
        "CNT_FAM_MEMBERS": np.random.randint(1, 7, n),
        "CODE_GENDER": np.random.choice(["M", "F"], n),
        "FLAG_OWN_CAR": np.random.choice(["Y", "N"], n),
        "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], n),
    })


def test_fix_anomalies(sample_df):
    sample_df.loc[0, "DAYS_EMPLOYED"] = 365243
    result = fix_anomalies(sample_df.copy())
    assert pd.isna(result.loc[0, "DAYS_EMPLOYED"])
    assert (result["DAYS_EMPLOYED"] != 365243).all()


def test_impute_missing(sample_df):
    sample_df.loc[:10, "AMT_INCOME_TOTAL"] = np.nan
    result = impute_missing(sample_df.copy())
    assert result["AMT_INCOME_TOTAL"].isna().sum() == 0


def test_encode_categoricals(sample_df):
    result = encode_categoricals(sample_df.copy())
    assert "CODE_GENDER" not in result.columns or result["CODE_GENDER"].dtype in [float, int]


def test_build_features(sample_df):
    result = build_features(sample_df.copy())
    assert "AGE_YEARS" in result.columns
    assert "EMPLOYMENT_YEARS" in result.columns
    assert "CREDIT_INCOME_RATIO" in result.columns
    assert "EXT_SOURCE_MEAN" in result.columns
    assert result["AGE_YEARS"].between(0, 100).all()


def test_assign_stage(sample_df):
    df_with_ext = sample_df.copy()
    df_with_ext["EXT_SOURCE_MEAN"] = np.random.uniform(0, 1, len(sample_df))
    result = assign_stage(df_with_ext)
    assert "IFRS9_STAGE" in result.columns
    assert result["IFRS9_STAGE"].isin([1, 2, 3]).all()
    # All TARGET=1 should be Stage 3
    assert (result[result["TARGET"] == 1]["IFRS9_STAGE"] == 3).all()


def test_compute_ecl(sample_df):
    df = sample_df.copy()
    df["EXT_SOURCE_MEAN"] = np.random.uniform(0, 1, len(df))
    result = compute_ecl(df)
    assert "ECL" in result.columns
    assert (result["ECL"] >= 0).all()


def test_simulate_mpesa():
    df = generate_borrower_transactions(borrower_id=0, n_months=3)
    assert len(df) > 0
    assert "borrower_id" in df.columns
    assert "amount" in df.columns
    assert "description" in df.columns
    assert "transaction_type" in df.columns
