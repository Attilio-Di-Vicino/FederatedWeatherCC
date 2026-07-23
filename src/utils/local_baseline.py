"""
local_baseline.py

Trains one independent model per station (no federation, no data sharing).
This is the LOCAL baseline required by reviewers:
  - If FL > local baseline → FL is justified
  - If local baseline > FL → federation hurts due to non-IID heterogeneity

Usage (run from project root):
    python src/utils/local_baseline.py --model transformer
    python src/utils/local_baseline.py --model crossformer

Results are saved to:
    data/trained_model/local_baseline_{model}_results.csv
    data/trained_model/local_baseline_{model}_ws{id}.pth  (per-station checkpoints)
"""

from __future__ import annotations

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "src/utils"))
from dataset import WeatherDataset

TEMP_RANGE = 45.0   # Celsius range after F→C and validity clamp: (0, 45)


def _load_model_module(model_name: str):
    sys.path.insert(0, os.path.join(ROOT, f"src/model/{model_name}"))
    import importlib
    train_mod = importlib.import_module("train")
    return train_mod


def train_local_model(
    station_id: int,
    df_station: pd.DataFrame,
    train_mod,
    config: dict,
    device: torch.device,
    model_name: str,
) -> dict:
    """Train one model for a single station and return test metrics."""
    input_window  = config["data"]["input_window"]
    output_window = config["data"]["output_window"]
    feature_cols  = config["data"]["feature_cols"]
    target_cols   = config["data"]["target_cols"]
    batch_size    = config["training"]["batch_size"]
    lr            = config["training"]["learning_rate"]
    epochs        = config["training"]["epochs"]
    es_patience   = int(config["training"].get("early_stopping_patience", 20))

    n         = len(df_station)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    train_ds = WeatherDataset(df_station, input_window, output_window,
                               feature_cols, target_cols, 0, train_end)
    val_ds   = WeatherDataset(df_station, input_window, output_window,
                               feature_cols, target_cols, train_end, val_end)
    test_ds  = WeatherDataset(df_station, input_window, output_window,
                               feature_cols, target_cols, val_end)

    if len(train_ds) == 0 or len(test_ds) == 0:
        print(f"  [ws{station_id}] Insufficient data — skipping.")
        return {}

    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    model     = train_mod.build_model(config, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    print(f"  [ws{station_id}] {len(train_ds)} train windows, "
          f"{len(test_ds)} test windows")

    for epoch in range(1, epochs + 1):
        if model_name == "transformer":
            tr_loss = train_mod.train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                output_window, len(feature_cols)
            )
            va_loss = train_mod.validate_one_epoch(
                model, val_loader, criterion, device,
                output_window, len(feature_cols)
            )
        else:
            tr_loss = train_mod.train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            va_loss = train_mod.validate_one_epoch(
                model, val_loader, criterion, device
            )

        import math
        if not math.isnan(va_loss):
            scheduler.step(va_loss)
            if va_loss < best_val - 1e-5:
                best_val   = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

        if epoch % 10 == 0:
            print(f"    epoch {epoch:>3}/{epochs}  train={tr_loss:.5f}  "
                  f"val={va_loss:.5f}  best={best_val:.5f}")

        if no_improve >= es_patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_dir = os.path.join(ROOT, "data/trained_model")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(ckpt_dir, f"local_baseline_{model_name}_ws{station_id}.pth")
    )

    # Evaluate
    if model_name == "transformer":
        results = train_mod.evaluate(
            model, test_loader, device,
            output_window, len(feature_cols), target_cols
        )
    else:
        results = train_mod.evaluate(model, test_loader, device, target_cols)

    r = results[0]
    r["station_id"] = station_id
    r["n_train"]    = len(train_ds)
    r["n_test"]     = len(test_ds)
    print(f"  [ws{station_id}] MAE={r['mae']:.4f} ({r['mae_celsius']:.2f}°C)  "
          f"RMSE={r['rmse']:.4f} ({r['rmse_celsius']:.2f}°C)  "
          f"Corr={r['correlation']:.4f}  "
          f"Skill(pers)={r['skill_persist']:.4f}")
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "crossformer"],
                        default="crossformer")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: src/model/<model>/config.yaml)")
    args = parser.parse_args()

    model_name = args.model
    config_path = args.config or os.path.join(
        ROOT, f"src/model/{model_name}/config.yaml"
    )

    # Change directory so relative paths in config work
    os.chdir(os.path.join(ROOT, f"src/model/{model_name}"))

    train_mod = _load_model_module(model_name)
    config    = train_mod.load_config(config_path)

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    csv_path = config["data"]["csv_path"]
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    station_ids = sorted(df["station_id"].unique())

    print(f"\nLocal Baseline — {model_name.upper()}  |  device={device}")
    print(f"Stations: {station_ids}")
    print("=" * 60)

    all_results = []
    for sid in station_ids:
        print(f"\nStation ws{sid}:")
        df_s = df[df["station_id"] == sid].sort_values("datetime").reset_index(drop=True)
        r = train_local_model(sid, df_s, train_mod, config, device, model_name)
        if r:
            all_results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print(f"LOCAL BASELINE SUMMARY — {model_name.upper()}")
    print("=" * 60)
    metrics = ["mae_celsius", "rmse_celsius", "bias_celsius", "correlation",
               "skill_zero", "skill_persist"]
    header = f"{'Station':<10}" + "".join(f"{m:>16}" for m in metrics)
    print(header)
    print("-" * len(header))
    for r in all_results:
        row = f"  ws{r['station_id']:<7}"
        for m in metrics:
            row += f"{r.get(m, float('nan')):>16.4f}"
        print(row)
    print("-" * len(header))

    # Aggregate mean
    agg = {}
    for m in metrics:
        vals = [r[m] for r in all_results if m in r]
        agg[m] = np.mean(vals)
    row = f"  {'MEAN':<8}"
    for m in metrics:
        row += f"{agg[m]:>16.4f}"
    print(row)

    # Save to CSV
    results_df = pd.DataFrame(all_results)
    out_path = os.path.join(ROOT, f"data/trained_model/local_baseline_{model_name}_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()