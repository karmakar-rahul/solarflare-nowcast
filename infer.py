import torch
import numpy as np
from src.model import SolarFlareMLP

THRESHOLD = 0.3   # high-recall early warning


def load_model(checkpoint_path: str):
    device = torch.device("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = SolarFlareMLP().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model


def predict(model, x: np.ndarray):
    """
    x shape: (360, 2)
    """
    assert x.shape == (360, 2)

    x = torch.from_numpy(x).float().unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    return {
        "probability": prob,
        "flare_warning": prob >= THRESHOLD
    }


if __name__ == "__main__":
    model = load_model("checkpoints/best_model.pt")
    print("Model loaded successfully on CPU.")

    # Dummy sample
    x = np.random.rand(360, 2).astype(np.float32)

    result = predict(model, x)
    print(result)
