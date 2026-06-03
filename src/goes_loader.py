"""
src/goes_loader.py
------------------
Loads a user-uploaded GOES CSV and returns a (360, 5) numpy array
ready for model inference — identical feature engineering to training.

The user's CSV must have at least two flux columns (short and long channel).
We handle the common naming variants from NOAA / SunPy exports.
"""

import numpy as np
import pandas as pd

SEQ_LEN      = 360
LOG_EPS      = 1e-9
ROLLING_MAX_W = 30

# Accepted column name variants for each channel
SHORT_CHANNEL_NAMES = {"xrs_short", "xrsa_flux", "a_flux", "flux_short", "xrsa"}
LONG_CHANNEL_NAMES  = {"xrs_long",  "xrsb_flux", "b_flux", "flux_long",  "xrsb"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename flux columns to canonical flux_short / flux_long."""
    rename = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for alias in SHORT_CHANNEL_NAMES:
        if alias in lower_cols:
            rename[lower_cols[alias]] = "flux_short"
            break

    for alias in LONG_CHANNEL_NAMES:
        if alias in lower_cols:
            rename[lower_cols[alias]] = "flux_long"
            break

    df = df.rename(columns=rename)

    if "flux_short" not in df.columns or "flux_long" not in df.columns:
        raise ValueError(
            "Could not find X-ray flux columns in the uploaded CSV.\n"
            f"Expected one of {SHORT_CHANNEL_NAMES} for the short channel and "
            f"{LONG_CHANNEL_NAMES} for the long channel.\n"
            f"Found: {df.columns.tolist()}"
        )
    return df


def load_goes_csv(file) -> tuple:
    """
    Load a user-supplied GOES CSV and return the feature array + DataFrame.

    Args:
        file : file path string or file-like object (e.g. Streamlit UploadedFile)

    Returns:
        (x_array, df)
            x_array : np.ndarray of shape (360, 5)  ready for inference
            df      : pd.DataFrame with time_tag + all feature columns (for display)
    """
    df = pd.read_csv(file)
    df = _normalise_columns(df)

    if len(df) < SEQ_LEN:
        raise ValueError(
            f"CSV must contain at least {SEQ_LEN} rows. Found {len(df)}."
        )

    df = df.tail(SEQ_LEN).reset_index(drop=True)

    # Clip to avoid log10(0)
    df["flux_short"] = pd.to_numeric(df["flux_short"], errors="coerce").clip(lower=LOG_EPS)
    df["flux_long"]  = pd.to_numeric(df["flux_long"],  errors="coerce").clip(lower=LOG_EPS)
    df.fillna(method="ffill", inplace=True)
    df.fillna(LOG_EPS, inplace=True)

    # Engineer features 
    df["xrs_short"]   = np.log10(df["flux_short"])
    df["xrs_long"]    = np.log10(df["flux_long"])
    df["xrs_ratio"]   = df["xrs_long"] - df["xrs_short"]
    df["deriv_short"] = df["xrs_short"].diff().fillna(0.0)
    df["rolling_max"] = df["xrs_long"].rolling(window=ROLLING_MAX_W, min_periods=1).max()

    feature_cols = ["xrs_short", "xrs_long", "xrs_ratio", "deriv_short", "rolling_max"]
    x_array = df[feature_cols].values.astype(np.float32)

    return x_array, df