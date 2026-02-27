"""
Data cleaning and feature engineering for stockout prediction.
"""

import numpy as np
import pandas as pd

from ingest import normalize_columns


FEATURE_COLUMNS = [
    "Closing Stock",
    "Opening Stock",
    "QTY transacted",
    "is_decrement",
    "rolling_qty_5",
    "days_since_last_txn",
    "month",
    "dayofweek",
]


def _add_next_stockout_date(item_df, stockout_level=0):
    """
    Add the next stockout date per item.

    Args:
        item_df (pd.DataFrame): Single-item dataframe sorted by date.
        stockout_level (float): Closing stock level to treat as stockout.

    Returns:
        pd.DataFrame: Item dataframe with next_stockout_date.
    """
    next_date = pd.NaT
    next_dates = []
    for closing, date in zip(
        reversed(item_df["Closing Stock"].tolist()),
        reversed(item_df["Transaction Date"].tolist()),
    ):
        if pd.notna(closing) and closing <= stockout_level:
            next_date = date
        next_dates.append(next_date)
    item_df = item_df.copy()
    item_df["next_stockout_date"] = list(reversed(next_dates))
    return item_df


def clean_and_engineer(df, stockout_level=0):
    """
    Clean and engineer features for modeling.

    Args:
        df (pd.DataFrame): Raw input dataframe.
        stockout_level (float): Closing stock level to treat as stockout.

    Returns:
        pd.DataFrame: Cleaned dataframe with engineered features.
    """
    model_df = normalize_columns(df.copy())

    model_df["Transaction Date"] = pd.to_datetime(
        model_df["Transaction Date"], errors="coerce"
    )
    model_df["Closing Stock"] = pd.to_numeric(model_df["Closing Stock"], errors="coerce")
    model_df["Opening Stock"] = pd.to_numeric(model_df["Opening Stock"], errors="coerce")
    model_df["QTY transacted"] = pd.to_numeric(
        model_df["QTY transacted"], errors="coerce"
    )

    model_df = model_df.sort_values(["Item name", "Transaction Date"])

    model_df = model_df.groupby("Item name", group_keys=False).apply(
        _add_next_stockout_date, stockout_level=stockout_level
    )
    model_df["days_to_stockout"] = (
        model_df["next_stockout_date"] - model_df["Transaction Date"]
    ).dt.days

    model_df["is_decrement"] = model_df["Type"].isin(
        ["Decrement", "Adjust Decrement"]
    ).astype(int)
    model_df["abs_qty"] = model_df["QTY transacted"].abs()
    model_df["rolling_qty_5"] = (
        model_df.groupby("Item name")["abs_qty"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    model_df["days_since_last_txn"] = (
        model_df.groupby("Item name")["Transaction Date"].diff().dt.days
    )
    model_df["month"] = model_df["Transaction Date"].dt.month
    model_df["dayofweek"] = model_df["Transaction Date"].dt.dayofweek

    return model_df


def build_train_features(df, horizon_days=30, stockout_level=0):
    """
    Build training features and targets.

    Args:
        df (pd.DataFrame): Raw input dataframe.
        horizon_days (int): Stockout horizon in days.
        stockout_level (float): Closing stock level to treat as stockout.

    Returns:
        tuple: (X, y_stockout, model_df)
    """
    model_df = clean_and_engineer(df, stockout_level=stockout_level)
    model_df = model_df[model_df["Closing Stock"] > 0].copy()

    model_df["stockout_in_horizon"] = (
        model_df["days_to_stockout"].notna()
        & (model_df["days_to_stockout"] <= horizon_days)
    ).astype(int)

    feature_df = model_df[FEATURE_COLUMNS + ["stockout_in_horizon"]].copy()
    for col in FEATURE_COLUMNS:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["stockout_in_horizon"]

    return X, y, model_df


def build_inference_features(df, stockout_level=0):
    """
    Build inference features for the latest record per item.

    Args:
        df (pd.DataFrame): Raw input dataframe.
        stockout_level (float): Closing stock level to treat as stockout.

    Returns:
        tuple: (X_latest, latest_df)
    """
    model_df = clean_and_engineer(df, stockout_level=stockout_level)
    model_df = model_df[model_df["Closing Stock"] > 0].copy()

    latest_df = (
        model_df.sort_values(["Item name", "Transaction Date"])
        .groupby("Item name")
        .tail(1)
        .copy()
    )

    feature_df = latest_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    return feature_df, latest_df
