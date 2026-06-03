"""
infer.py
--------
CLI inference script for the Solar Flare Early Warning model.

Usage:
    # Run on live GOES data:
    python infer.py --live

    # Run on a local GOES CSV:
    python infer.py --csv path/to/goes_data.csv

    # Specify a different checkpoint:
    python infer.py --live --checkpoint checkpoints/best_model.pt
"""

import argparse
import numpy as np
import torch

from src.model import build_model
from src.predictor import FlarePredictor
from src.metrics import find_best_threshold


DEFAULT_CHECKPOINT = "checkpoints/best_model.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Solar Flare Early Warning — inference")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    p.add_argument("--live",  action="store_true", help="Fetch live GOES data from NOAA SWPC")
    p.add_argument("--csv",   type=str, default=None, help="Path to a local GOES CSV file")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold override (default: use checkpoint value)")
    return p.parse_args()


def load_predictor(checkpoint_path: str, threshold_override: float = None) -> FlarePredictor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = FlarePredictor.from_checkpoint(checkpoint_path, device=device)
    if threshold_override is not None:
        predictor.threshold = threshold_override
        print(f"[infer] Threshold overridden to {threshold_override:.2f}")
    print(f"[infer] Model loaded from {checkpoint_path}  "
          f"(device={device}, threshold={predictor.threshold:.2f})")
    return predictor


def run_inference(predictor: FlarePredictor, x: np.ndarray):
    """Run a single prediction and print a formatted result."""
    prob, warning = predictor.predict(x)
    level, emoji  = predictor.get_risk_level(prob)

    print("\n" + "=" * 50)
    print("  SOLAR FLARE EARLY WARNING SYSTEM")
    print("=" * 50)
    print(f"  Flare probability (next 60 min) : {prob:.3f}")
    print(f"  Risk level                      : {emoji} {level}")
    print(f"  Warning issued                  : {'⚠ YES' if warning else '✓ NO'}")
    print("=" * 50)

    if warning:
        print("\n  ⚠  Elevated X-ray flux detected.")
        print("     Monitor NOAA SWPC for official alerts:")
        print("     https://www.swpc.noaa.gov/\n")
    else:
        print("\n  ✓  No immediate flare risk detected.\n")


if __name__ == "__main__":
    args = parse_args()
    predictor = load_predictor(args.checkpoint, args.threshold)

    if args.live:
        print("[infer] Fetching live GOES data from NOAA SWPC...")
        from src.goes_fetcher import fetch_latest_goes_array
        x = fetch_latest_goes_array()
        print(f"[infer] Live data loaded: shape={x.shape}")
        run_inference(predictor, x)

    elif args.csv:
        print(f"[infer] Loading GOES CSV: {args.csv}")
        from src.goes_loader import load_goes_csv
        x, _ = load_goes_csv(args.csv)
        print(f"[infer] CSV loaded: shape={x.shape}")
        run_inference(predictor, x)

    else:
        print("[infer] No input source specified. Using random dummy data for model check.")
        x = np.random.rand(360, 5).astype(np.float32)
        # Scale to realistic log-flux range (~A to M class background)
        x[:, 0] = np.random.uniform(-7, -4, 360).astype(np.float32)   # xrs_short
        x[:, 1] = np.random.uniform(-6, -3, 360).astype(np.float32)   # xrs_long
        x[:, 2] = x[:, 1] - x[:, 0]                                    # ratio
        x[:, 3] = np.diff(x[:, 0], prepend=x[0, 0]).astype(np.float32) # derivative
        x[:, 4] = x[:, 1]                                               # rolling max approx
        run_inference(predictor, x)
        print("[infer] Tip: run with --live or --csv for real data.")