"""
Train and save the Gradient Boosting stockout model.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from ingest import read_csv_file
from prepare import build_train_features, FEATURE_COLUMNS


def train_model(
    df,
    threshold=0.10,
    artifacts_dir="artifacts",
    horizon_days=30,
    stockout_level=0,
):
    """
    Train the Gradient Boosting model and save artifacts.

    Args:
        df (pd.DataFrame): Raw input dataframe.
        threshold (float): Decision threshold for stockout classification.
        artifacts_dir (str): Directory to save model artifacts.
        horizon_days (int): Stockout horizon in days.
        stockout_level (float): Closing stock level to treat as stockout.

    Returns:
        dict: Training metrics and artifact paths.
    """
    X, y, _ = build_train_features(
        df, horizon_days=horizon_days, stockout_level=stockout_level
    )

    class_counts = y.value_counts(dropna=False).to_dict()
    print(f"Label distribution: {class_counts}")

    if y.nunique() < 2:
        raise ValueError(
            "Training labels have only one class. "
            "Increase horizon_days or provide data with stockouts."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.8,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "threshold": float(threshold),
    }

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / "stockout_gb_model.joblib"
    meta_path = artifacts_path / "stockout_model_meta.json"

    joblib.dump(model, model_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_columns": FEATURE_COLUMNS, **metrics}, f, indent=2)

    return {
        "metrics": metrics,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
    }


def _parse_args():
    """
    Parse command-line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train stockout model.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Decision threshold for stockout classification",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=30,
        help="Stockout horizon in days",
    )
    parser.add_argument(
        "--stockout-level",
        type=float,
        default=0,
        help="Closing stock level to treat as stockout (e.g., 0 or 5)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to save model artifacts",
    )
    return parser.parse_args()


def _load_df(csv_path):
    """
    Load input dataframe from CSV.

    Args:
        csv_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    return read_csv_file(csv_path)


if __name__ == "__main__":
    args = _parse_args()
    df = _load_df(args.csv)
    result = train_model(
        df,
        threshold=args.threshold,
        artifacts_dir=args.artifacts_dir,
        horizon_days=args.horizon_days,
        stockout_level=args.stockout_level,
    )
    print("Training complete.")
    print(json.dumps(result["metrics"], indent=2))
