import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


GOES_XRS_URL = (
    "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
)


def fetch_latest_goes_csv() -> pd.DataFrame:
    """
    Fetches the latest GOES X-ray data and returns
    the last 360 minutes as a DataFrame.
    """

    response = requests.get(GOES_XRS_URL, timeout=15)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)

    required_cols = ["time_tag", "energy", "flux"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError("Unexpected GOES data format")

    # Separate short and long channels
    short_df = df[df["energy"] == "0.05-0.4nm"]
    long_df = df[df["energy"] == "0.1-0.8nm"]

    # Convert timestamps
    short_df["time_tag"] = pd.to_datetime(short_df["time_tag"])
    long_df["time_tag"] = pd.to_datetime(long_df["time_tag"])

    # Merge on time
    merged = pd.merge(
        short_df[["time_tag", "flux"]],
        long_df[["time_tag", "flux"]],
        on="time_tag",
        suffixes=("_short", "_long")
    )

    merged = merged.sort_values("time_tag")

    if len(merged) < 360:
        raise ValueError("Not enough recent GOES data available")

    latest = merged.tail(360).copy()

    # Log scaling (training-consistent)
    latest["xrs_short"] = np.log10(latest["flux_short"] + 1e-8)
    latest["xrs_long"] = np.log10(latest["flux_long"] + 1e-8)

    return latest[["time_tag", "xrs_short", "xrs_long"]]
