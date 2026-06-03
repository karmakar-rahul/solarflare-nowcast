"""
src/model.py
------------
Solar Flare Early Warning — Model Architecture

Replaces the original flat MLP (which destroyed all temporal ordering) with a
1D-CNN + LSTM hybrid that actually understands trends in the X-ray flux signal.

Architecture:
  Input  : (batch, 360, 5)  — 360 minutes × 5 channels
             [xrs_short, xrs_long, xrs_ratio, derivative_short, rolling_max_long]
  CNN    : Two Conv1d blocks extract local temporal patterns (gradients, peaks)
  LSTM   : Single LSTM layer captures longer-range sequential dependencies
  Head   : Dropout → Linear(1) binary output (logit)

Why this beats the old Linear(720, 512) MLP:
  - Preserves temporal ordering (minute 1 ≠ minute 360)
  - Conv layers detect rising-flux gradients (key precursor signal)
  - LSTM remembers multi-hour trends
  - Fits comfortably in 4 GB VRAM (RTX 2050)
"""

import torch
import torch.nn as nn


class SolarFlareCNN(nn.Module):
    """
    1D-CNN + LSTM binary solar flare classifier.

    Input shape  : (batch_size, seq_len, n_features)  e.g. (32, 360, 5)
    Output shape : (batch_size, 1)  raw logit (apply sigmoid for probability)
    """

    def __init__(
        self,
        n_features: int = 5,
        seq_len: int = 360,
        cnn_channels: tuple = (32, 64),
        kernel_size: int = 7,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # CNN feature extractor 
        # Input comes in as (batch, seq_len, n_features); permute to
        # (batch, n_features, seq_len) for Conv1d
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(n_features, cnn_channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),                      # seq_len → 180

            # Block 2
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),                      # 180 → 90
        )

        # LSTM temporal aggregator 
        # After pooling, we have (batch, cnn_channels[1], seq/4)
        cnn_out_len = seq_len // 4                # 360 // 4 = 90
        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        #  Classification head 
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),                    # raw logit
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        returns : (batch, 1) raw logit
        """
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)           # → (B, n_features, 360)
        x = self.conv_block(x)            # → (B, 64, 90)

        # LSTM expects (batch, seq, features)
        x = x.permute(0, 2, 1)           # → (B, 90, 64)
        _, (h_n, _) = self.lstm(x)        # h_n: (1, B, lstm_hidden)
        x = h_n[-1]                       # → (B, lstm_hidden)

        return self.head(x)               # → (B, 1)


# Convenience factory 
def build_model(
    n_features: int = 5,
    seq_len: int = 360,
    dropout: float = 0.3,
) -> SolarFlareCNN:
    """Return a freshly initialised SolarFlareCNN."""
    return SolarFlareCNN(
        n_features=n_features,
        seq_len=seq_len,
        dropout=dropout,
    )

if __name__ == "__main__":
    # Quick sanity check
    model = build_model()
    dummy = torch.randn(4, 360, 5)
    out = model(dummy)
    print("Output shape:", out.shape)   # expect (4, 1)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")