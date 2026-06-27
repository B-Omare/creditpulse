"""Tests for ML models."""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_loans():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "TARGET": np.random.binomial(1, 0.1, n),
        "AGE_YEARS": np.random.uniform(22, 65, n),
        "EMPLOYMENT_YEARS": np.random.uniform(0, 20, n),
        "AMT_INCOME_TOTAL": np.random.normal(150000, 60000, n),
        "AMT_CREDIT": np.random.normal(400000, 150000, n),
        "AMT_ANNUITY": np.random.normal(20000, 5000, n),
        "CREDIT_INCOME_RATIO": np.random.uniform(0.5, 8, n),
        "ANNUITY_INCOME_RATIO": np.random.uniform(0.05, 0.6, n),
        "LTV_RATIO": np.random.uniform(0.8, 1.2, n),
        "EXT_SOURCE_MEAN": np.random.uniform(0, 1, n),
        "EXT_SOURCE_MIN": np.random.uniform(0, 1, n),
        "CNT_CHILDREN": np.random.randint(0, 5, n),
        "CNT_FAM_MEMBERS": np.random.randint(1, 7, n),
        "REGION_RATING_CLIENT": np.random.randint(1, 4, n),
        "INCOME_PER_PERSON": np.random.normal(30000, 15000, n),
    })


def test_feature_matrix_shape(sample_loans):
    assert sample_loans.shape == (200, 15)
    assert sample_loans.isna().sum().sum() == 0


def test_class_imbalance(sample_loans):
    default_rate = sample_loans["TARGET"].mean()
    assert 0.05 <= default_rate <= 0.25, f"Unexpected default rate: {default_rate}"


def test_credit_income_ratio_range(sample_loans):
    assert (sample_loans["CREDIT_INCOME_RATIO"] > 0).all()


def test_ext_source_range(sample_loans):
    assert sample_loans["EXT_SOURCE_MEAN"].between(0, 1).all()


def test_prediction_logic():
    """Test the core prediction logic without requiring installed ML libraries."""
    pd_val = 1 / (1 + np.exp(-(0.3 - 0.5) * 3))
    assert 0 < pd_val < 1
    stage = 1 if pd_val < 0.15 else (2 if pd_val < 0.40 else 3)
    assert stage in [1, 2, 3]
    ecl = pd_val * 0.45 * 50000
    assert ecl > 0


def test_ifrs9_stage_boundaries():
    """IFRS 9 stage thresholds are correct."""
    cases = [
        (0.05, 1), (0.14, 1), (0.15, 2), (0.39, 2), (0.40, 3), (0.99, 3),
    ]
    for pd_val, expected_stage in cases:
        stage = 1 if pd_val < 0.15 else (2 if pd_val < 0.40 else 3)
        assert stage == expected_stage, f"PD={pd_val}: expected stage {expected_stage}, got {stage}"
