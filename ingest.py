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
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

    return df
