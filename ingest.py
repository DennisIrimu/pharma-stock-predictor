"""
Simple ingestion utilities for stockout pipeline.
"""

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "Item name",
    "Transaction Date",
    "Opening Stock",
    "QTY transacted",
    "Closing Stock",
    "Type",
    "Sales value",
}

def _normalize_col(col):
    """
    Normalize column names for matching.

    Args:
        col (str): Column name.

    Returns:
        str: Normalized column key.
    """
    cleaned = str(col).replace("\ufeff", "").replace("\xa0", " ")
    cleaned = cleaned.replace("_", " ").strip().lower()
    cleaned = " ".join(cleaned.split())
    return cleaned


def normalize_columns(df):
    """
    Normalize dataframe columns to expected names when possible.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with normalized column names.
    """
    canonical_map = {_normalize_col(c): c for c in REQUIRED_COLUMNS}
    rename_map = {}
    for col in df.columns:
        key = _normalize_col(col)
        if key in canonical_map:
            rename_map[col] = canonical_map[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def read_csv_file(file_path):
    """
    Read a CSV file and validate required columns.

    Args:
        file_path (str | Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df = normalize_columns(df)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

    return df
