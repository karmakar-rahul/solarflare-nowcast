"""
src/dataset.py
--------------
Dataset construction for solar flare binary prediction.

Key design decisions vs the old approach:
  1. CORRECT LABELS  — Labels come from matching the flare catalog
                        (flare_catalog.csv) against the flux timeline.
                        A window is positive (y=1) if ANY M/X flare peak
                        falls within HORIZON_MINUTES after the window end.
                        Previously, labels were phase tags (RISING/PEAK/…)
                        which only detected an ongoing flare, not a future one.

  2. 5 FEATURE CHANNELS instead of 1:
        xrs_short, xrs_long, xrs_ratio, deriv_short, rolling_max
     These are the same channels used by goes_fetcher.py for live inference.

  3. ON-THE-FLY WINDOWING via a torch Dataset — avoids pre-materialising
     all (200k × 360 × 5) windows in RAM (~11 GB). Each __getitem__ slices
     only what it needs.

  4. TRAIN / VAL / TEST SPLIT done chronologically (not random) to prevent
     data leakage across time.

Usage:
    from src.dataset import build_datasets
    train_ds, val_ds, test_ds = build_datasets(
        flux_csv="goes_xray.csv",
        flare_csv="flare_catalog.csv",
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Constants 

SEQ_LEN          = 360    # 6-hour look-back window (minutes)
HORIZON_MINUTES  = 60     # forecast horizon — "will a flare occur in next 60 min?"
LOG_EPS          = 1e-9   # floor before log10 to avoid -inf
ROLLING_MAX_W    = 30     # rolling-max window in minutes


# Feature engineering 

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 5 feature channels from a raw GOES flux DataFrame.

    Expected input columns: time (datetime), flux_short (W/m²), flux_long (W/m²)
    Added columns         : xrs_short, xrs_long, xrs_ratio, deriv_short, rolling_max
    """
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["xrs_short"]   = np.log10(df["flux_short"].clip(lower=LOG_EPS))
    df["xrs_long"]    = np.log10(df["flux_long"].clip(lower=LOG_EPS))
    df["xrs_ratio"]   = df["xrs_long"] - df["xrs_short"]
    df["deriv_short"] = df["xrs_short"].diff().fillna(0.0)
    df["rolling_max"] = df["xrs_long"].rolling(window=ROLLING_MAX_W, min_periods=1).max()

    return df


FEATURE_COLS = ["xrs_short", "xrs_long", "xrs_ratio", "deriv_short", "rolling_max"]


#  Label construction 

def build_labels(flux_df: pd.DataFrame, flare_df: pd.DataFrame) -> np.ndarray:
    """
    For each minute in flux_df, assign label = 1 if any M/X flare peak
    occurs within the next HORIZON_MINUTES minutes.

    Args:
        flux_df  : DataFrame with a 'time' column (UTC datetime, 1-min cadence)
        flare_df : DataFrame with a 'peak_time' column (UTC datetime)

    Returns:
        np.ndarray of shape (len(flux_df),) with int labels {0, 1}
    """
    times  = flux_df["time"].values.astype("datetime64[ns]")
    labels = np.zeros(len(times), dtype=np.int8)

    horizon = np.timedelta64(HORIZON_MINUTES, "m")

    for peak in flare_df["peak_time"].values.astype("datetime64[ns]"):
        # Mark all windows whose end time is within [peak - horizon, peak]
        mask = (times >= peak - horizon) & (times <= peak)
        labels[mask] = 1

    pos_rate = labels.mean() * 100
    print(f"[dataset] Labels built: {labels.sum():,} positive / {len(labels):,} total "
          f"({pos_rate:.2f}% positive)")
    return labels


# torch Dataset 

class GoesFlareDataset(Dataset):
    """
    Sliding-window dataset over GOES X-ray flux.

    Each sample is:
        X : float32 array of shape (SEQ_LEN, 5)
        y : float32 scalar {0.0, 1.0}

    The dataset skips the first SEQ_LEN rows (not enough history)
    and the last HORIZON_MINUTES rows (future labels would be unknown
    at inference time).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        features : (N, 5) float32 — pre-computed feature matrix
        labels   : (N,)   int8   — binary labels
        """
        assert len(features) == len(labels), "features and labels length mismatch"
        self.features = features.astype(np.float32)
        self.labels   = labels.astype(np.float32)
        # Valid indices: we need SEQ_LEN history, skip last HORIZON_MINUTES
        self.start = SEQ_LEN
        self.end   = len(features) - HORIZON_MINUTES
        if self.end <= self.start:
            raise ValueError("Dataset too short for the chosen SEQ_LEN and HORIZON_MINUTES.")

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        i = self.start + idx
        x = self.features[i - SEQ_LEN : i]    # (360, 5)
        y = self.labels[i]                      # scalar
        return x, y


# Data loading helpers 
def _load_flux_csv(path: str) -> pd.DataFrame:
    """
    Load a GOES X-ray CSV.  Handles both the raw netCDF-converted format
    (columns: time, xrsa_flux, xrsb_flux) and the pre-processed format
    (columns: time_tag, flux_short, flux_long).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise column names
    rename = {}
    for c in df.columns:
        if c in ("xrsa_flux", "a_flux", "flux_short"):
            rename[c] = "flux_short"
        elif c in ("xrsb_flux", "b_flux", "flux_long"):
            rename[c] = "flux_long"
        elif c in ("time_tag", "time", "datetime", "timestamp"):
            rename[c] = "time"
    df.rename(columns=rename, inplace=True)

    # Handle CSVs that are already log-scaled with xrs_short / xrs_long column names.
    # In this case we reverse the log10 to recover raw flux so the feature engineering
    # pipeline (which re-applies log10) stays consistent.
    already_log = False
    if "flux_short" not in df.columns and "xrs_short" in df.columns:
        df.rename(columns={"xrs_short": "flux_short", "xrs_long": "flux_long"}, inplace=True)
        already_log = True   # values are log10(flux); we'll undo below

    required = {"time", "flux_short", "flux_long"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{path}' is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # If values were already log-scaled, convert back to linear flux
    # so that engineer_features() can re-apply log10 uniformly.
    if already_log:
        df["flux_short"] = 10.0 ** df["flux_short"].clip(lower=-9)
        df["flux_long"]  = 10.0 ** df["flux_long"].clip(lower=-9)

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df.dropna(subset=["time", "flux_short", "flux_long"], inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_flare_csv(path: str) -> pd.DataFrame:
    """
    Load a flare catalog CSV with at least a 'peak_time' column.
    Only M/X class flares are retained.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={"class": "flare_class"}, inplace=True)

    df["peak_time"] = pd.to_datetime(df["peak_time"], utc=True, errors="coerce")
    df.dropna(subset=["peak_time"], inplace=True)

    # Keep only M/X class (significant flares)
    if "flare_class" in df.columns:
        df = df[df["flare_class"].str.upper().str[0].isin(["M", "X"])].copy()

    df.sort_values("peak_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[dataset] Loaded {len(df)} M/X flare events from catalog.")
    return df


# Main public API 
def build_datasets(
    flux_csv: str,
    flare_csv: str,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    # test_frac is the remainder
) -> tuple:
    """
    Build chronological train / val / test splits.

    Returns:
        (train_dataset, val_dataset, test_dataset, pos_weight)
        pos_weight is a float you can use to initialise WeightedBCELoss.
    """
    # 1. Load raw data
    flux_df  = _load_flux_csv(flux_csv)
    flare_df = _load_flare_csv(flare_csv)

    print(f"[dataset] Flux data: {len(flux_df):,} rows  "
          f"({flux_df['time'].min()} → {flux_df['time'].max()})")

    # 2. Engineer features
    flux_df = engineer_features(flux_df)
    features = flux_df[FEATURE_COLS].values.astype(np.float32)

    # 3. Build binary labels
    labels = build_labels(flux_df, flare_df)

    # 4. Chronological split (no shuffle — avoids data leakage across time)
    n = len(features)
    i_train = int(n * train_frac)
    i_val   = int(n * (train_frac + val_frac))

    train_ds = GoesFlareDataset(features[:i_train],  labels[:i_train])
    val_ds   = GoesFlareDataset(features[i_train:i_val], labels[i_train:i_val])
    test_ds  = GoesFlareDataset(features[i_val:],    labels[i_val:])

    print(f"[dataset] Splits → train: {len(train_ds):,}  "
          f"val: {len(val_ds):,}  test: {len(test_ds):,}")

    # 5. Compute pos_weight for WeightedBCELoss (neg_count / pos_count)
    n_pos = int(labels[:i_train].sum())
    n_neg = i_train - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"[dataset] Train pos_weight (neg/pos): {pos_weight:.1f}")

    return train_ds, val_ds, test_ds, pos_weight


if __name__ == "__main__":
    # Quick smoke test — update paths to your actual files
    train_ds, val_ds, test_ds, pw = build_datasets(
        flux_csv="../solarflare_goes_data/goes_xray.csv",
        flare_csv="../solarflare_goes_data/flare_catalog.csv",
    )
    x, y = train_ds[0]
    print("Sample X shape:", x.shape, "  y:", y)