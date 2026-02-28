"""Shared utilities for loading LinkedIn export CSVs and detecting the user."""

import glob
import os
from collections import Counter

import pandas as pd


def find_csv_files(folder: str) -> dict[str, str]:
    """Find connections and messages CSV files in the given folder.

    Returns dict with keys 'connections' and 'messages' mapping to file paths.
    Raises FileNotFoundError if either file is not found.
    """
    folder = os.path.expanduser(folder.strip().strip("'\""))
    result = {}

    for pattern, key in [("*onnection*.csv", "connections"), ("*essage*.csv", "messages")]:
        matches = glob.glob(os.path.join(folder, pattern))
        if not matches:
            raise FileNotFoundError(f"No {key} CSV found in {folder} (pattern: {pattern})")
        result[key] = matches[0]

    return result


def load_connections_csv(path: str) -> pd.DataFrame:
    """Load LinkedIn connections CSV, skipping the metadata header rows.

    LinkedIn exports have 3 lines of metadata before the actual header row.
    Falls back to scanning for the 'First Name' header if the default skip doesn't work.
    """
    try:
        df = pd.read_csv(path, skiprows=3)
        if "First Name" in df.columns:
            return df
    except Exception:
        pass

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "First Name" in line:
                return pd.read_csv(path, skiprows=i)

    raise ValueError(f"Could not find 'First Name' header in {path}")


def load_messages_csv(path: str) -> pd.DataFrame:
    """Load LinkedIn messages CSV and parse DATE column as UTC datetime."""
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], utc=True)
    return df


def detect_user_url(messages_df: pd.DataFrame, sample_size: int = 20) -> str:
    """Detect the logged-in user's profile URL from message data.

    Uses the most frequently appearing URL in sender/recipient columns
    across the first `sample_size` rows.
    """
    sample = messages_df.head(sample_size)
    urls = []
    for col in ["SENDER PROFILE URL", "RECIPIENT PROFILE URLS"]:
        if col in sample.columns:
            urls.extend(sample[col].dropna().tolist())

    counter = Counter(urls)
    if not counter:
        raise ValueError("Could not detect user URL from messages")

    return counter.most_common(1)[0][0]
