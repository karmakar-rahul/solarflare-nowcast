import numpy as np
import torch
from collections import deque


class FlarePredictor:
    """
    High-recall binary solar flare early-warning predictor.
    """

    def __init__(
        self,
        model,
        history_size=5,
        threshold=0.55
    ):
        self.model = model
        self.history = deque(maxlen=history_size)
        self.threshold = threshold

    def predict(self, x: np.ndarray):
        x_tensor = torch.from_numpy(x).float().unsqueeze(0)

        with torch.no_grad():
            logit = self.model(x_tensor).item()

        # Raw sigmoid probability (no temperature scaling)
        prob = torch.sigmoid(torch.tensor(logit)).item()

        self.history.append(prob)
        smoothed = float(np.mean(self.history))

        warning = smoothed >= self.threshold
        return smoothed, warning

    def get_history(self):
        return list(self.history)
