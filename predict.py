"""
Inference utilities for stockout prediction.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from prepare import build_inference_features, FEATURE_COLUMNS


def load_artifacts(artifacts_dir="artifacts"):
    """
    Load model and metadata artifacts.

    Args:
        artifacts_dir (str): Directory containing saved artifacts.

    Returns:
        tuple: (model, metadata dict)
    """
    artifacts_path = Path(artifacts_dir)
    model_path = artifacts_path / "stockout_gb_model.joblib"
    meta_path = artifacts_path / "stockout_model_meta.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Model artifacts not found. Run train first.")

    model = joblib.load(model_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, meta


def score_latest(df, artifacts_dir="artifacts", stockout_level=0):
    """
    Score the latest record per item and return recommendations.

    Args:
        df (pd.DataFrame): Raw input dataframe.
        artifacts_dir (str): Directory containing saved artifacts.
        stockout_level (float): Closing stock level to treat as stockout.

    Returns:
        pd.DataFrame: Recommendations dataframe.
    """
    model, meta = load_artifacts(artifacts_dir)
    threshold = float(meta.get("threshold", 0.10))

    X_latest, latest_df = build_inference_features(df, stockout_level=stockout_level)
    X_latest = X_latest.reindex(columns=FEATURE_COLUMNS)

    probs = model.predict_proba(X_latest)[:, 1]
    recs = (probs >= threshold).astype(int)

    output = pd.DataFrame(
        {
            "Item name": latest_df["Item name"].values,
            "Transaction Date": latest_df["Transaction Date"].values,
            "Closing Stock": latest_df["Closing Stock"].values,
            "stockout_prob_30d": probs,
            "recommendation": [
                "Reorder / Review" if r == 1 else "OK"
                for r in recs
            ],
        }
    ).sort_values(["recommendation", "stockout_prob_30d"], ascending=[False, False])

    return output
