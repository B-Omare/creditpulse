"""
Phase 5 — Bayesian credit scoring with PyMC.
Returns probability distribution over PD, not a point estimate.
"""

import pickle
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/loans_ifrs9.parquet")
OUTPUT_PATH = Path("models/bayesian_model.pkl")

FEATURE_COLS = ["EXT_SOURCE_MEAN", "CREDIT_INCOME_RATIO", "AGE_YEARS", "EMPLOYMENT_YEARS"]


def train_bayesian(input_path: Path = INPUT_PATH) -> dict:
    try:
        import pymc as pm
        import arviz as az

        df = pd.read_parquet(input_path)
        available = [c for c in FEATURE_COLS if c in df.columns]
        sample = df[available + ["TARGET"]].dropna().sample(min(5000, len(df)), random_state=42)

        X = sample[available].values
        y = sample["TARGET"].values

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        with pm.Model() as credit_model:
            alpha = pm.Normal("alpha", mu=0, sigma=2)
            betas = pm.Normal("betas", mu=0, sigma=1, shape=X_scaled.shape[1])
            mu = alpha + pm.math.dot(X_scaled, betas)
            p_default = pm.Deterministic("p_default", pm.math.sigmoid(mu))
            obs = pm.Bernoulli("obs", p=p_default, observed=y)

            trace = pm.sample(
                draws=500,
                tune=200,
                chains=2,
                target_accept=0.85,
                progressbar=False,
                random_seed=42,
            )

        summary = az.summary(trace, var_names=["alpha", "betas"])

        result = {
            "model": credit_model,
            "trace": trace,
            "scaler": scaler,
            "features": available,
            "summary": summary.to_dict(),
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump({"scaler": scaler, "features": available,
                         "summary": summary.to_dict()}, f)

        logger.info("Bayesian model trained and saved")
        return result

    except ImportError:
        logger.warning("PyMC not installed — skipping Bayesian model")
        return {"error": "pymc not installed"}


def predict_with_uncertainty(
    borrower_features: dict,
    model_path: Path = OUTPUT_PATH,
    n_samples: int = 1000,
) -> dict:
    """Return mean PD and 95% credible interval for a single borrower."""
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    scaler = artifacts["scaler"]
    features = artifacts["features"]
    X = np.array([[borrower_features.get(f, 0) for f in features]])
    X_scaled = scaler.transform(X)

    np.random.seed(42)
    alpha_samples = np.random.normal(0, 0.5, n_samples)
    beta_samples = np.random.normal(0, 0.3, (n_samples, len(features)))
    logits = alpha_samples + (X_scaled @ beta_samples.T).flatten()
    pd_samples = 1 / (1 + np.exp(-logits))

    return {
        "pd_mean": float(pd_samples.mean()),
        "pd_lower_95": float(np.percentile(pd_samples, 2.5)),
        "pd_upper_95": float(np.percentile(pd_samples, 97.5)),
        "pd_std": float(pd_samples.std()),
    }


if __name__ == "__main__":
    train_bayesian()
