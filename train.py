"""
train.py
--------
Training script for the Solar Flare Early Warning CNN+LSTM model.

Improvements over the old training approach:
  • Correct binary labels from flare catalog (not phase tags)
  • 5-channel feature input (xrs_short/long, ratio, derivative, rolling_max)
  • Focal loss for extreme class imbalance (~1-2% positive rate)
  • Mixed-precision training (torch.cuda.amp) for RTX 2050 4GB
  • TSS and HSS evaluation metrics (standard in space weather)
  • EarlyStopping on validation TSS (not just loss)
  • Checkpoint saves best model by TSS
  • All hyper-parameters in config.yaml

Run:
    python train.py
    python train.py --config config.yaml          # explicit config path
    python train.py --smoke                       # quick 2-epoch smoke test
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml

from tqdm import tqdm
from src.dataset import build_datasets
from src.focal_loss import FocalLoss, WeightedBCELoss
from src.model import build_model
from src.metrics import compute_tss_hss


#  Argument parsing 

def parse_args():
    p = argparse.ArgumentParser(description="Train Solar Flare Warning model")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("--smoke", action="store_true", help="2-epoch smoke test on small subset")
    p.add_argument("--resume", action="store_true", help="Resume training from checkpoints/last_model.pt")
    return p.parse_args()


# Evaluation 

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Returns dict with loss, accuracy, TSS, HSS, precision, recall.
    Uses BCE (not focal) for the eval loss so it stays comparable across runs.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_probs, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels).astype(int)
    preds      = (all_probs >= threshold).astype(int)

    metrics = compute_tss_hss(all_labels, preds)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# Main training loop 

def train(cfg: dict, args, smoke: bool = False):
    # Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    #  Datasets 
    data_cfg = cfg["data"]
    train_ds, val_ds, test_ds, pos_weight = build_datasets(
        flux_csv=data_cfg["flux_csv"],
        flare_csv=data_cfg["flare_csv"],
    )

    if smoke:
        # Tiny subset for quick iteration
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(2000, len(train_ds))))
        val_ds   = Subset(val_ds,   range(min(500,  len(val_ds))))
        print("[train] SMOKE MODE: using small subset")

    train_cfg = cfg["train"]
    batch_size = train_cfg["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
        persistent_workers=train_cfg.get("num_workers", 2) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model_cfg = cfg.get("model", {})
    model = build_model(
        n_features=model_cfg.get("n_features", 5),
        seq_len=model_cfg.get("seq_len", 360),
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable parameters: {total_params:,}")

    # Loss 
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "focal")

    if loss_type == "focal":
        criterion = FocalLoss(
            alpha=loss_cfg.get("alpha", 0.25),
            gamma=loss_cfg.get("gamma", 2.0),
        ).to(device)
        print(f"[train] Loss: FocalLoss(alpha={loss_cfg.get('alpha', 0.25)}, "
              f"gamma={loss_cfg.get('gamma', 2.0)})")
    else:
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        criterion = WeightedBCELoss(pos_weight=pw).to(device)
        print(f"[train] Loss: WeightedBCE(pos_weight={pos_weight:.1f})")

    #  Optimiser & schedulers 
    lr = train_cfg.get("learning_rate", 1e-3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Mixed precision 
    use_amp = device.type == "cuda" and train_cfg.get("mixed_precision", True)
    scaler  = GradScaler("cuda",enabled=use_amp)
    print(f"[train] Mixed precision (AMP): {use_amp}")
    # Resume from checkpoint
    start_epoch  = 1
    best_tss     = -1.0
    patience_ctr = 0

    resume_path = "checkpoints/last_model.pt"
    if args.resume and os.path.exists(resume_path):  
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("[train] No optimizer state found in checkpoint. Starting optimizer fresh.")
        start_epoch  = ckpt.get("epoch", 1) + 1
        best_tss     = ckpt.get("val_tss", -1.0)
        patience_ctr = ckpt.get("patience_ctr", 0)
        print(f"[train] Resumed from epoch {start_epoch - 1}  "
              f"(best TSS so far: {best_tss:.4f})")
    else:
        print("[train] Starting fresh training run")
    # Checkpoint directory 
    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/best_model.pt"
    last_path = "checkpoints/last_model.pt"

    # Training loop 
    epochs       = 2 if smoke else train_cfg.get("epochs", 50)
    patience     = train_cfg.get("patience", 8)
   

    print(f"\n[train] Starting training for up to {epochs} epochs "
          f"(early stop patience={patience})\n{'='*60}")

    history = {"train_loss": [], "val_loss": [], "val_tss": [], "val_hss": []}

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs}", unit="batch", leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda",enabled=use_amp):
                logits = model(x).squeeze(1)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        val_metrics    = evaluate(model, val_loader, device)
        val_tss        = val_metrics["tss"]
        val_hss        = val_metrics["hss"]

        scheduler.step(val_tss)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"lr={current_lr:.2e}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_TSS={val_tss:.4f}  "
            f"val_HSS={val_hss:.4f}  "
            f"recall={val_metrics['recall']:.4f}  "
            f"precision={val_metrics['precision']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_tss"].append(val_tss)
        history["val_hss"].append(val_hss)

        # Checkpoint best
        if val_tss > best_tss:
            best_tss = val_tss
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_tss": val_tss,
                    "val_hss": val_hss,
                    "patience_ctr": patience_ctr,
                    "cfg": cfg,
                },
                best_path,
            )
            print(f"  ✓ New best TSS={best_tss:.4f} → saved to {best_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_tss": best_tss,
                    "patience_ctr": patience_ctr,
                    "cfg": cfg,
                },
                last_path,)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n[train] Early stopping at epoch {epoch} "
                      f"(no TSS improvement for {patience} epochs)")
                break

    # Save last checkpoint
    torch.save({"epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_tss": best_tss,
        "patience_ctr": patience_ctr,
        "cfg": cfg,}, last_path)

    # Test set evaluation
    print("\n[train] Loading best model for final test evaluation...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=train_cfg.get("num_workers", 2),
    )
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("=" * 60)

    # Save training history 
    import json
    with open("checkpoints/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[train] Training history saved to checkpoints/training_history.json")


# Entry point 

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg, args, smoke=args.smoke)
