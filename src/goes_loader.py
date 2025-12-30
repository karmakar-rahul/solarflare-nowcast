import pandas as pd
import numpy as np


REQUIRED_COLUMNS = ["xrs_short", "xrs_long"]


def load_goes_csv(file) -> np.ndarray:
    """
    Loads a GOES CSV and returns the last 360 minutes as (360, 2).
    """

    df = pd.read_csv(file)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if len(df) < 360:
        raise ValueError("CSV must contain at least 360 rows")

    df = df.tail(360)

    xrs_short = df["xrs_short"].values
    xrs_long = df["xrs_long"].values

    # simple log normalization (robust for GOES flux)
    xrs_short = np.log10(xrs_short + 1e-8)
    xrs_long = np.log10(xrs_long + 1e-8)

    return np.stack([xrs_short, xrs_long], axis=1)
