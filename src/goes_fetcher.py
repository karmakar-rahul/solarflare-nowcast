"""
src/goes_fetcher.py
-------------------
Fetches live GOES X-ray flux from NOAA SWPC and returns the last 360 minutes
as a DataFrame with all 5 engineered feature channels.

Feature channels (must match training feature engineering in dataset.py):
  0  xrs_short     log10(0.05–0.4 nm flux + 1e-9)
  1  xrs_long      log10(0.1–0.8 nm flux  + 1e-9)
  2  xrs_ratio     xrs_long - xrs_short  (log-space ratio; hardens near flare)
  3  deriv_short   1-minute finite difference of xrs_short (rising slope)
  4  rolling_max   rolling max of xrs_long over last 30 minutes
"""

import numpy as np
import pandas as pd
import requests

GOES_XRS_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

# Epsilon added before log10 to avoid -inf on near-zero flux values.
# 1e-9 W/m² corresponds to a quiet-sun floor well below A-class flares.
LOG_EPS = 1e-9

# Window sizes (in minutes = rows, since data is 1-minute cadence)
SEQ_LEN       = 360   # 6-hour input window
ROLLING_MAX_W = 30    # 30-minute rolling max


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a merged DataFrame with columns flux_short and flux_long,
    compute all 5 feature channels in-place and return the result.
    """
    df = df.copy()

    # Log-scale the raw flux
    df["xrs_short"] = np.log10(df["flux_short"].clip(lower=LOG_EPS))
    df["xrs_long"]  = np.log10(df["flux_long"].clip(lower=LOG_EPS))

    # Log-space ratio (increases during impulsive phase)
    df["xrs_ratio"] = df["xrs_long"] - df["xrs_short"]

    # 1-minute derivative of the short channel (first to respond)
    df["deriv_short"] = df["xrs_short"].diff().fillna(0.0)

    # 30-minute rolling max of long channel (captures preceding peak activity)
    df["rolling_max"] = (
        df["xrs_long"]
        .rolling(window=ROLLING_MAX_W, min_periods=1)
        .max()
    )

    return df


def fetch_latest_goes_df() -> pd.DataFrame:
    """
    Fetches the latest GOES X-ray data and returns the last SEQ_LEN minutes
    as a tidy DataFrame with columns:
        time_tag, flux_short, flux_long, xrs_short, xrs_long,
        xrs_ratio, deriv_short, rolling_max

    Raises:
        requests.HTTPError   if the NOAA endpoint returns an error
        ValueError           if there are fewer than SEQ_LEN data points
    """
    response = requests.get(GOES_XRS_URL, timeout=15)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)

    required_cols = {"time_tag", "energy", "flux"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Unexpected GOES data format. Got columns: {df.columns.tolist()}")

    # Split short and long channels
    short_df = df[df["energy"] == "0.05-0.4nm"][["time_tag", "flux"]].copy()
    long_df  = df[df["energy"] == "0.1-0.8nm"][["time_tag", "flux"]].copy()

    # Parse timestamps
    for d in (short_df, long_df):
        d["time_tag"] = pd.to_datetime(d["time_tag"], utc=True)

    # Merge on time
    merged = pd.merge(
        short_df.rename(columns={"flux": "flux_short"}),
        long_df.rename(columns={"flux": "flux_long"}),
        on="time_tag",
    ).sort_values("time_tag").reset_index(drop=True)

    if len(merged) < SEQ_LEN:
        raise ValueError(
            f"Only {len(merged)} data points available; need at least {SEQ_LEN}."
        )

    # Take the most recent SEQ_LEN minutes
    merged = merged.tail(SEQ_LEN).reset_index(drop=True)

    # Engineer features
    merged = _engineer_features(merged)

    return merged


def fetch_latest_goes_array() -> np.ndarray:
    """
    Convenience wrapper that returns the live data as a numpy array
    of shape (SEQ_LEN, 5) ready for model inference.
    """
    df = fetch_latest_goes_df()
    feature_cols = ["xrs_short", "xrs_long", "xrs_ratio", "deriv_short", "rolling_max"]
    return df[feature_cols].values.astype(np.float32)