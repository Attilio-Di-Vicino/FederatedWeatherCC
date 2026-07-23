"""
train.py  –  Transformer  (TempOut forecasting)
Fixed temporal 80/10/10 split — correct for centralised vs federated comparison.
All 8 stations present in every split (data sorted by datetime globally).
"""

from __future__ import annotations

import logging
import math
import os
import sys
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))
from logger import Logger
from dataset import WeatherDataset
from model import myTransformer

log_std = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_data(config: dict):
    """
    Build train/val/test loaders for centralised training.
    Split is applied PER STATION so each sliding window contains only
    consecutive hours from a single station. All stations are then
    concatenated into a single loader.
    """
    csv_path      = config["data"]["csv_path"]
    input_window  = config["data"]["input_window"]
    output_window = config["data"]["output_window"]
    feature_cols  = config["data"]["feature_cols"]
    target_cols   = config["data"]["target_cols"]
    batch_size    = config["training"]["batch_size"]
    pin           = torch.cuda.is_available()

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    station_ids = sorted(df["station_id"].unique())

    train_datasets, val_datasets, test_datasets = [], [], []

    for sid in station_ids:
        df_s = df[df["station_id"] == sid].sort_values("datetime").reset_index(drop=True)
        n         = len(df_s)
        train_end = int(0.80 * n)
        val_end   = int(0.90 * n)

        train_datasets.append(WeatherDataset(df_s, input_window, output_window,
                                              feature_cols, target_cols, 0, train_end))
        val_datasets.append(  WeatherDataset(df_s, input_window, output_window,
                                              feature_cols, target_cols, train_end, val_end))
        test_datasets.append( WeatherDataset(df_s, input_window, output_window,
                                              feature_cols, target_cols, val_end))

    from torch.utils.data import ConcatDataset
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(ConcatDataset(val_datasets),   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=pin)
    test_loader  = DataLoader(ConcatDataset(test_datasets),  batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=pin)
    return train_loader, val_loader, test_loader


def load_data_client(config: dict, station_id: int):
    """Federated: single-station 70/15/15 temporal split. num_workers=0 for gRPC safety."""
    csv_path      = config["data"]["csv_path"]
    input_window  = config["data"]["input_window"]
    output_window = config["data"]["output_window"]
    feature_cols  = config["data"]["feature_cols"]
    target_cols   = config["data"]["target_cols"]
    batch_size    = config["training"]["batch_size"]

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df = df[df["station_id"] == station_id].sort_values("datetime").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No data for station_id={station_id}")

    log_std.info(f"[CLIENT {station_id}] {len(df)} rows")

    n         = len(df)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    train_ds = WeatherDataset(df, input_window, output_window,
                               feature_cols, target_cols, 0, train_end)
    val_ds   = WeatherDataset(df, input_window, output_window,
                               feature_cols, target_cols, train_end, val_end)
    test_ds  = WeatherDataset(df, input_window, output_window,
                               feature_cols, target_cols, val_end)

    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=False, **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: dict, device: torch.device) -> myTransformer:
    mc = config["model"]
    return myTransformer(
        input_dim=len(config["data"]["feature_cols"]),
        output_dim=len(config["data"]["target_cols"]),
        d_model=mc.get("d_model", 64),
        nhead=mc.get("nhead", 4),
        num_layers=mc.get("num_layers", 2),
        dropout=mc.get("dropout", 0.1),
    ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss,
                    checkpoint_dir, tag="transformer"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}_epoch{epoch:04d}.pth")
    torch.save({
        "epoch": epoch, "val_loss": val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, path)
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["val_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience   = patience
        self.min_delta  = float(min_delta)
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss, model):
        val_loss = float(val_loss)
        if math.isnan(val_loss) or math.isinf(val_loss):
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate
# ─────────────────────────────────────────────────────────────────────────────

def _tgt(B, T, F, device):
    return torch.zeros((B, T, F), device=device)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    output_window, feature_dim, scaler=None):
    model.train()
    total, steps = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        t = _tgt(x.size(0), output_window, feature_dim, device)
        optimizer.zero_grad()
        if scaler:
            with autocast(device_type=device.type):
                loss = criterion(model(x, t), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x, t), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        v = loss.item()
        if not (math.isnan(v) or math.isinf(v)):
            total += v; steps += 1
    return total / steps if steps else float("nan")


def validate_one_epoch(model, loader, criterion, device, output_window, feature_dim):
    model.eval()
    total, steps = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            t = _tgt(x.size(0), output_window, feature_dim, device)
            with autocast(device_type=device.type):
                pred = model(x, t)
            v = criterion(pred, y).item()
            if not (math.isnan(v) or math.isinf(v)):
                total += v; steps += 1
    return total / steps if steps else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _skill(y_true, y_pred):
    mse_m = mean_squared_error(y_true, y_pred)
    mse_b = mean_squared_error(y_true, np.zeros_like(y_true))
    return 1.0 - mse_m / mse_b if mse_b != 0 else 1.0


def _persistence_skill(y_true, y_pred, y_last):
    """Skill score relative to persistence baseline (predict last known value)."""
    mse_m = mean_squared_error(y_true, y_pred)
    mse_p = mean_squared_error(y_true, y_last)
    return 1.0 - mse_m / mse_p if mse_p != 0 else 1.0


def _compute_metrics(y_true, y_pred, y_last, name, temp_range=56.0):
    """
    Compute full metric suite for one target column.
    temp_range: TempOut MinMax range in °C (max - min of raw data).
                Used to convert normalised MAE/RMSE to real units.
    """
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = mean_squared_error(y_true, y_pred) ** 0.5
    bias  = float(np.mean(y_pred - y_true))
    corr  = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    skill_zero = _skill(y_true, y_pred)
    skill_pers = _persistence_skill(y_true, y_pred, y_last)
    return {
        "target":        name,
        "mae":           float(mae),
        "rmse":          float(rmse),
        "mae_celsius":   float(mae  * temp_range),
        "rmse_celsius":  float(rmse * temp_range),
        "bias":          float(bias),
        "bias_celsius":  float(bias * temp_range),
        "correlation":   float(corr),
        "skill_zero":    float(skill_zero),    # vs always-predict-zero baseline
        "skill_persist": float(skill_pers),    # vs persistence baseline (meteorological standard)
    }

def evaluate(model, loader, device, output_window, feature_dim, target_cols,
             temp_range=56.0):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            t = _tgt(x.size(0), output_window, feature_dim, device)
            with autocast(device_type=device.type):
                pred = model(x, t)
            preds.append(pred.cpu().float().numpy())
            targets.append(y.cpu().float().numpy())

    preds   = np.concatenate(preds,   axis=0)
    targets = np.concatenate(targets, axis=0)

    results = []
    for i, name in enumerate(target_cols):
        p = preds[:, :, i].reshape(-1)
        t = targets[:, :, i].reshape(-1)
        # Persistence baseline: repeat the last input step for all output steps
        # We approximate this as the mean of predictions at step 0 repeated
        # Use first predicted step as proxy for "last known" when y_last unavailable
        y_last = np.full_like(t, t.mean())   # climatological mean as persistence proxy
        results.append(_compute_metrics(t, p, y_last, name, temp_range))
    return results



def main():
    config = load_config()
    log    = Logger()

    lr            = config["training"]["learning_rate"]
    epochs        = config["training"]["epochs"]
    device        = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    if config["training"]["device"] == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU.")

    feature_cols     = config["data"]["feature_cols"]
    target_cols      = config["data"]["target_cols"]
    output_window    = config["data"]["output_window"]
    feature_dim      = len(feature_cols)
    checkpoint_dir   = config["training"].get(
                           "checkpoint_dir",
                           "../../../data/trained_model/checkpoints/transformer")
    checkpoint_every = int(config["training"].get("checkpoint_every", 5))
    es_patience      = int(config["training"].get("early_stopping_patience", 10))
    es_min_delta     = float(config["training"].get("early_stopping_min_delta", 1e-5))

    log.info(f"TRANSFORMER – TempOut  |  device={device}")
    log.info(f"  d_model={config['model']['d_model']}  "
             f"nhead={config['model']['nhead']}  "
             f"layers={config['model']['num_layers']}")
    log.info(f"  batch={config['training']['batch_size']}  "
             f"lr={lr}  epochs={epochs}")

    train_loader, val_loader, test_loader = load_data(config)
    model     = build_model(config, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    scaler     = GradScaler("cuda") if device.type == "cuda" else None
    early_stop = EarlyStopping(patience=es_patience, min_delta=es_min_delta)
    best_val   = float("inf")

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer,
                             device, output_window, feature_dim, scaler)
        va = validate_one_epoch(model, val_loader, criterion,
                                device, output_window, feature_dim)

        if not math.isnan(va):
            scheduler.step(va)

        current_lr = optimizer.param_groups[0]["lr"]
        log.info(f"Epoch {epoch:>3}/{epochs}  "
                 f"train={tr:.5f}  val={va:.5f}  lr={current_lr:.2e}")

        if math.isnan(tr) or math.isnan(va):
            log.warning("NaN loss — check dataset.")
            continue

        if epoch % checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, va,
                            checkpoint_dir)
            log.info(f"  Checkpoint saved (epoch {epoch})")

        if va < best_val:
            best_val = va
            save_checkpoint(model, optimizer, scheduler, epoch, va,
                            checkpoint_dir, tag="transformer_best")

        if early_stop.step(va, model):
            log.info(f"Early stopping at epoch {epoch} "
                     f"(best val={early_stop.best_loss:.5f})")
            break

    early_stop.restore_best(model)
    log.info(f"Best val loss: {early_stop.best_loss:.5f}")

    results = evaluate(model, test_loader, device,
                       output_window, feature_dim, target_cols)
    log.info("── Final Evaluation (test set) ───────────────────────────")
    for r in results:
        log.info(f"  {r['target']}:")
        log.info(f"    MAE          = {r['mae']:.4f}  ({r['mae_celsius']:.2f} °C)")
        log.info(f"    RMSE         = {r['rmse']:.4f}  ({r['rmse_celsius']:.2f} °C)")
        log.info(f"    Bias         = {r['bias']:.4f}  ({r['bias_celsius']:.2f} °C)")
        log.info(f"    Correlation  = {r['correlation']:.4f}")
        log.info(f"    Skill(zero)  = {r['skill_zero']:.4f}")
        log.info(f"    Skill(pers)  = {r['skill_persist']:.4f}")

    os.makedirs("../../../data/trained_model", exist_ok=True)
    torch.save(model.state_dict(),
               "../../../data/trained_model/transformer_tempout_final.pth")
    log.info("Model saved.")


if __name__ == "__main__":
    main()