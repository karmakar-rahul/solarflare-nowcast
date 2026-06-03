"""
src/predictor.py
----------------
High-recall binary solar flare early-warning predictor.

Changes from original:
  • Accepts (360, 5) input arrays (5 feature channels, not 2)
  • Loads threshold from checkpoint (set during training via find_best_threshold)
    rather than hard-coding 0.55
  • Temporal smoothing window is configurable
  • get_risk_level() maps probability → human-readable alert level
"""

import numpy as np
import torch
from collections import deque


RISK_LEVELS = [
    (0.75, "HIGH"),
    (0.50, "ELEVATED"),
    (0.25, "MODERATE"),
    (0.00, "LOW"),
]


class FlarePredictor:
    """
    Wraps the trained SolarFlareCNN for single-step or streaming inference.

    Args:
        model         : loaded SolarFlareCNN in eval mode
        threshold     : decision threshold for binary warning (default 0.5)
        history_size  : temporal smoothing window over consecutive predictions
        device        : torch device string ('cpu' or 'cuda')
    """

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        history_size: int = 5,
        device: str = "cpu",
    ):
        self.model      = model
        self.threshold  = threshold
        self.history    = deque(maxlen=history_size)
        self.device     = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_class=None, device: str = "cpu"):
        """
        Load a predictor directly from a checkpoint file.

        The checkpoint must contain 'model_state' and optionally 'threshold'.
        """
        if model_class is None:
            from src.model import build_model
            model_class = build_model

        ckpt = torch.load(checkpoint_path, map_location=device)

        # Support both a plain state_dict and a full checkpoint dict
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            cfg = ckpt.get("cfg", {})
            model_cfg = cfg.get("model", {})
            model = model_class(
                n_features=model_cfg.get("n_features", 5),
                seq_len=model_cfg.get("seq_len", 360),
                dropout=0.0,           # no dropout at inference
            )
            model.load_state_dict(ckpt["model_state"])
            threshold = ckpt.get("threshold", 0.5)
        else:
            # Legacy: bare state_dict
            from src.model import SolarFlareCNN
            model = SolarFlareCNN()
            model.load_state_dict(ckpt)
            threshold = 0.5

        return cls(model, threshold=threshold, device=device)

    # Core inference 

    def predict_raw(self, x: np.ndarray) -> float:
        """
        Single forward pass.

        Args:
            x : np.ndarray of shape (360, 5) — one 6-hour window

        Returns:
            float probability in [0, 1]
        """
        if x.shape != (360, 5):
            raise ValueError(f"Expected input shape (360, 5), got {x.shape}")

        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x_tensor).squeeze().item()

        return float(torch.sigmoid(torch.tensor(logit)).item())

    def predict(self, x: np.ndarray) -> tuple:
        """
        Predict with temporal smoothing.

        Args:
            x : np.ndarray of shape (360, 5)

        Returns:
            (smoothed_prob, flare_warning)
                smoothed_prob  : float — running mean of last `history_size` raw probs
                flare_warning  : bool  — True if smoothed_prob >= threshold
        """
        raw_prob = self.predict_raw(x)
        self.history.append(raw_prob)
        smoothed = float(np.mean(self.history))
        warning  = smoothed >= self.threshold
        return smoothed, warning

    def get_risk_level(self, prob: float) -> tuple:
        """
        Map a probability to a human-readable risk level.

        Returns:
            (label, emoji)  e.g. ("HIGH")
        """
        for threshold, label, emoji in RISK_LEVELS:
            if prob >= threshold:
                return label, emoji
        return "LOW"

    def reset_history(self):
        """Clear the smoothing buffer (call when switching data sources)."""
        self.history.clear()

    def get_history(self):
        return list(self.history)