"""
per_station_eval.py

Loads the per-station FL client checkpoints saved during federated training
and evaluates each one on its local test set.

This produces the per-station FL results table requested by reviewers,
enabling direct comparison with:
  - centralised model (single global model, all data pooled)
  - local baseline (one independent model per station)
  - federated model (per-station checkpoint from last FL round)

Usage (run from project root):
    python src/utils/per_station_eval.py --model transformer
    python src/utils/per_station_eval.py --model crossformer

Outputs:
    data/trained_model/fl_per_station_{model}_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "src/utils"))
from dataset import WeatherDataset

TEMP_RANGE = 45.0   # Celsius range after F→C and validity clamp: (0, 45)


def _load_model_module(model_name: str):
    sys.path.insert(0, os.path.join(ROOT, f"src/model/{model_name}"))
    import importlib
    return importlib.import_module("train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "crossformer"],
                        default="crossformer")
    args = parser.parse_args()

    model_name  = args.model
    config_path = os.path.join(ROOT, f"src/model/{model_name}/config.yaml")
    os.chdir(os.path.join(ROOT, f"src/model/{model_name}"))

    train_mod = _load_model_module(model_name)
    config    = train_mod.load_config(config_path)

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    csv_path    = config["data"]["csv_path"]
    feature_cols = config["data"]["feature_cols"]
    target_cols  = config["data"]["target_cols"]
    input_window  = config["data"]["input_window"]
    output_window = config["data"]["output_window"]
    batch_size    = config["training"]["batch_size"]

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    station_ids = sorted(df["station_id"].unique())

    print(f"\nPer-Station FL Evaluation — {model_name.upper()}  |  device={device}")
    print("=" * 70)

    all_results = []
    for sid in station_ids:
        ckpt_path = os.path.join(
            ROOT, f"data/trained_model/client_{sid}_{model_name}.pth"
        )
        if not os.path.exists(ckpt_path):
            print(f"  [ws{sid}] Checkpoint not found: {ckpt_path} — skipping.")
            continue

        df_s = df[df["station_id"] == sid].sort_values("datetime").reset_index(drop=True)
        n       = len(df_s)
        val_end = int(0.85 * n)

        test_ds = WeatherDataset(df_s, input_window, output_window,
                                  feature_cols, target_cols, val_end)
        if len(test_ds) == 0:
            print(f"  [ws{sid}] No test windows — skipping.")
            continue

        test_loader = DataLoader(test_ds, batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)

        # Load model and checkpoint
        model = train_mod.build_model(config, device)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)

        # Evaluate
        if model_name == "transformer":
            results = train_mod.evaluate(
                model, test_loader, device,
                output_window, len(feature_cols), target_cols
            )
        else:
            results = train_mod.evaluate(model, test_loader, device, target_cols)

        r = results[0]
        r["station_id"] = sid
        r["n_test"]     = len(test_ds)
        all_results.append(r)

        print(f"  [ws{sid}]  n_test={len(test_ds):>5}  "
              f"MAE={r['mae']:.4f} ({r['mae_celsius']:.2f}°C)  "
              f"RMSE={r['rmse']:.4f} ({r['rmse_celsius']:.2f}°C)  "
              f"Corr={r['correlation']:.4f}  "
              f"Skill(pers)={r['skill_persist']:.4f}")

    if not all_results:
        print("No results — run FL training first.")
        return

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY — {model_name.upper()} FL (per-station checkpoints)")
    print("=" * 70)
    metrics = ["mae_celsius", "rmse_celsius", "bias_celsius",
               "correlation", "skill_zero", "skill_persist"]
    header = f"{'Station':<10}" + "".join(f"{m:>18}" for m in metrics)
    print(header)
    print("-" * len(header))
    for r in all_results:
        row = f"  ws{r['station_id']:<7}"
        for m in metrics:
            row += f"{r.get(m, float('nan')):>18.4f}"
        print(row)
    print("-" * len(header))

    agg_row = f"  {'MEAN':<8}"
    for m in metrics:
        vals = [r[m] for r in all_results if m in r]
        agg_row += f"{np.mean(vals):>18.4f}"
    print(agg_row)

    # Save
    results_df = pd.DataFrame(all_results)
    out_path = os.path.join(
        ROOT, f"data/trained_model/fl_per_station_{model_name}_results.csv"
    )
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()