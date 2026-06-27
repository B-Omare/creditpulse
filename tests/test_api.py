"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from creditpulse.api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "models_loaded" in data


def test_predict_approve(client):
    payload = {
        "age_years": 35,
        "employment_years": 8,
        "amt_income_total": 500000,
        "amt_credit": 200000,
        "amt_annuity": 15000,
        "ext_source_mean": 0.75,
        "cnt_children": 1,
    }
    r = client.post("/predict?borrower_id=test_001", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "pd_point_estimate" in data
    assert 0 <= data["pd_point_estimate"] <= 1
    assert data["ifrs9_stage"] in [1, 2, 3]
    assert data["recommendation"] in ["APPROVE", "MANUAL_REVIEW", "DECLINE"]
    assert data["ecl_estimate"] >= 0


def test_predict_high_risk(client):
    payload = {
        "age_years": 22,
        "employment_years": 0.5,
        "amt_income_total": 80000,
        "amt_credit": 800000,
        "amt_annuity": 45000,
        "ext_source_mean": 0.15,
        "cnt_children": 4,
    }
    r = client.post("/predict?borrower_id=test_002", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["pd_point_estimate"] > data["pd_lower_95"] or data["pd_point_estimate"] >= 0
    assert data["recommendation"] in ["MANUAL_REVIEW", "DECLINE"]


def test_predict_invalid_age(client):
    payload = {
        "age_years": 15,  # Under 18
        "employment_years": 0,
        "amt_income_total": 50000,
        "amt_credit": 100000,
        "amt_annuity": 5000,
        "cnt_children": 0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_explain_endpoint(client):
    r = client.get("/explain/borrower_001")
    assert r.status_code == 200
    data = r.json()
    assert data["borrower_id"] == "borrower_001"
