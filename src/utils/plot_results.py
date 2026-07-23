"""
plot_results.py

Generates all figures requested by reviewers:
  1. FL learning curves (MAE and RMSE vs round) for both models
  2. Predicted vs Observed scatter plots (centralised model, test set)
  3. Time series comparison: observed vs predicted (one station, test period)
  4. Per-station bar chart: centralised vs local vs federated MAE

Usage (run from project root):
    python src/utils/plot_results.py --model crossformer
    python src/utils/plot_results.py --model transformer
    python src/utils/plot_results.py --model both

Output directory: data/figures/
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")   # no display needed — save to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIG_DIR = os.path.join(ROOT, "data/figures")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src/utils"))
from dataset import WeatherDataset

TEMP_RANGE = 45.0   # Celsius range after F→C and validity clamp: (0, 45)
STYLE_PARAMS = {
    "figure.dpi":        150,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
}
plt.rcParams.update(STYLE_PARAMS)

COLORS = {
    "centralised": "#2196F3",
    "federated":   "#FF9800",
    "local":       "#4CAF50",
    "observed":    "#333333",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load model module
# ─────────────────────────────────────────────────────────────────────────────

def _load_module(model_name: str):
    sys.path.insert(0, os.path.join(ROOT, f"src/model/{model_name}"))
    import importlib
    return importlib.import_module("train")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — FL learning curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_fl_curves(fl_rounds_data: dict, model_name: str):
    """
    fl_rounds_data: dict with keys "rounds", "mae", "rmse" (lists, one per round).
    Plots MAE and RMSE vs FL round number.
    """
    rounds = fl_rounds_data["rounds"]
    mae    = [v * TEMP_RANGE for v in fl_rounds_data["mae"]]
    rmse   = [v * TEMP_RANGE for v in fl_rounds_data["rmse"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Federated Learning Convergence — {model_name.capitalize()}",
                 fontweight="bold")

    for ax, values, ylabel, title in zip(
        axes,
        [mae, rmse],
        ["MAE (°C)", "RMSE (°C)"],
        ["Mean Absolute Error vs Round", "Root Mean Squared Error vs Round"],
    ):
        ax.plot(rounds, values, marker="o", markersize=4,
                color=COLORS["federated"], linewidth=1.8, label="FL (FedProx)")
        ax.set_xlabel("FL Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out = os.path.join(FIG_DIR, f"fl_curves_{model_name}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Predicted vs Observed scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_obs(model, loader, device, model_name: str,
                     output_window: int, feature_dim: int, label: str):
    from torch.amp import autocast
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if model_name == "transformer":
                t    = torch.zeros((x.size(0), output_window, feature_dim), device=device)
                pred = model(x, t)
            else:
                pred = model(x)
            preds.append(pred.cpu().float().numpy())
            targets.append(y.cpu().float().numpy())

    p = np.concatenate(preds,   axis=0)[:, :, 0].reshape(-1) * TEMP_RANGE
    t = np.concatenate(targets, axis=0)[:, :, 0].reshape(-1) * TEMP_RANGE

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, alpha=0.15, s=3, color=COLORS["centralised"])
    lim = [min(t.min(), p.min()) - 1, max(t.max(), p.max()) + 1]
    ax.plot(lim, lim, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Observed TempOut (°C)")
    ax.set_ylabel("Predicted TempOut (°C)")
    ax.set_title(f"Predicted vs Observed — {label}")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend()
    ax.grid(True, alpha=0.3)

    corr = float(np.corrcoef(t, p)[0, 1])
    mae  = float(np.mean(np.abs(t - p)))
    ax.text(0.05, 0.92, f"r = {corr:.3f}\nMAE = {mae:.2f}°C",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    tag = label.lower().replace(" ", "_")
    out = os.path.join(FIG_DIR, f"pred_vs_obs_{tag}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Time series: observed vs predicted (one station)
# ─────────────────────────────────────────────────────────────────────────────

def plot_time_series(model, df_station: pd.DataFrame, config: dict,
                     device: torch.device, model_name: str, station_id: int,
                     label: str):
    from torch.amp import autocast

    feature_cols  = config["data"]["feature_cols"]
    target_cols   = config["data"]["target_cols"]
    input_window  = config["data"]["input_window"]
    output_window = config["data"]["output_window"]

    n       = len(df_station)
    val_end = int(0.85 * n)
    test_ds = WeatherDataset(df_station, input_window, output_window,
                              feature_cols, target_cols, val_end)
    loader  = DataLoader(test_ds, batch_size=64, shuffle=False,
                          num_workers=0, pin_memory=False)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if model_name == "transformer":
                t    = torch.zeros((x.size(0), output_window,
                                    len(feature_cols)), device=device)
                pred = model(x, t)
            else:
                pred = model(x)
            preds.append(pred[:, 0, 0].cpu().float().numpy())   # first output step
            targets.append(y[:, 0, 0].cpu().float().numpy())

    p = np.concatenate(preds)   * TEMP_RANGE
    t = np.concatenate(targets) * TEMP_RANGE

    # Use datetime index from test portion
    test_df = df_station.iloc[val_end + input_window: val_end + input_window + len(t)]
    x_axis  = test_df["datetime"].values if "datetime" in test_df.columns else np.arange(len(t))

    # Plot max 500 points for clarity
    n_plot = min(500, len(t))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_axis[:n_plot], t[:n_plot], color=COLORS["observed"],
            linewidth=1.0, label="Observed", alpha=0.9)
    ax.plot(x_axis[:n_plot], p[:n_plot], color=COLORS["centralised"],
            linewidth=1.0, label=f"Predicted ({label})", alpha=0.8, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("TempOut (°C)")
    ax.set_title(f"Observed vs Predicted — ws{station_id} — {label} (test set, first 500h)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    tag = label.lower().replace(" ", "_")
    out = os.path.join(FIG_DIR, f"timeseries_ws{station_id}_{tag}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Per-station comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_station_comparison(model_name: str):
    """
    Loads per-station result CSVs (centralised, local, FL) and plots
    a grouped bar chart of MAE in °C per station.
    """
    paths = {
        "Local Baseline": os.path.join(
            ROOT, f"data/trained_model/local_baseline_{model_name}_results.csv"),
        "Federated (FL)": os.path.join(
            ROOT, f"data/trained_model/fl_per_station_{model_name}_results.csv"),
    }

    available = {k: pd.read_csv(v) for k, v in paths.items() if os.path.exists(v)}
    if not available:
        print("No per-station result CSVs found — run local_baseline.py and "
              "per_station_eval.py first.")
        return

    station_ids = sorted(
        set.intersection(*[set(df["station_id"]) for df in available.values()])
    )
    x = np.arange(len(station_ids))
    width = 0.8 / len(available)
    colors_list = [COLORS["local"], COLORS["federated"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(available.items()):
        df_s = df[df["station_id"].isin(station_ids)].sort_values("station_id")
        mae  = df_s["mae_celsius"].values
        ax.bar(x + i * width - (len(available)-1)*width/2,
               mae, width=width*0.9, label=label,
               color=colors_list[i % len(colors_list)], alpha=0.85)

    ax.set_xlabel("Station")
    ax.set_ylabel("MAE (°C)")
    ax.set_title(f"Per-Station MAE Comparison — {model_name.capitalize()}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ws{s}" for s in station_ids])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"per_station_comparison_{model_name}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_for_model(model_name: str):
    config_path = os.path.join(ROOT, f"src/model/{model_name}/config.yaml")
    os.chdir(os.path.join(ROOT, f"src/model/{model_name}"))

    train_mod    = _load_module(model_name)
    config       = train_mod.load_config(config_path)
    feature_cols = config["data"]["feature_cols"]
    target_cols  = config["data"]["target_cols"]
    input_window  = config["data"]["input_window"]
    output_window = config["data"]["output_window"]

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    csv_path = config["data"]["csv_path"]
    df = pd.read_csv(csv_path, parse_dates=["datetime"])

    # ── Figure 2+3: centralised model ────────────────────────────────────────
    ckpt_central = os.path.join(
        ROOT, f"data/trained_model/{model_name}_tempout_final.pth"
    )
    if os.path.exists(ckpt_central):
        model = train_mod.build_model(config, device)
        ckpt_data = torch.load(ckpt_central, map_location="cpu")
        state_dict = ckpt_data.get("model", ckpt_data)
        model.load_state_dict(state_dict)
        model.to(device)

        # Build full test loader (all stations, per-station split → ConcatDataset)
        from torch.utils.data import ConcatDataset
        test_datasets = []
        for sid in sorted(df["station_id"].unique()):
            df_s = df[df["station_id"] == sid].sort_values("datetime").reset_index(drop=True)
            n = len(df_s)
            val_end = int(0.90 * n)
            ds = WeatherDataset(df_s, input_window, output_window,
                                 feature_cols, target_cols, val_end)
            if len(ds) > 0:
                test_datasets.append(ds)

        test_loader = DataLoader(
            ConcatDataset(test_datasets), batch_size=128,
            shuffle=False, num_workers=0, pin_memory=False
        )

        plot_pred_vs_obs(model, test_loader, device, model_name,
                         output_window, len(feature_cols),
                         f"{model_name.capitalize()} Centralised")

        # Time series for station 1
        df_s1 = df[df["station_id"] == 1].sort_values("datetime").reset_index(drop=True)
        plot_time_series(model, df_s1, config, device, model_name, 1,
                         f"{model_name.capitalize()} Centralised")
    else:
        print(f"Centralised checkpoint not found: {ckpt_central}")

    # ── Figure 4: per-station comparison ─────────────────────────────────────
    plot_per_station_comparison(model_name)

    print(f"\nAll figures saved to: {FIG_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "crossformer", "both"],
                        default="both")
    parser.add_argument(
        "--fl-rounds",
        type=str,
        default=None,
        help="Path to CSV with columns: round,mae,rmse (for FL learning curve plot). "
             "If not provided, skips the FL curve plot."
    )
    args = parser.parse_args()

    models = (["transformer", "crossformer"] if args.model == "both"
              else [args.model])

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Generating figures for: {model_name.upper()}")
        print(f"{'='*60}")
        run_for_model(model_name)

    # ── FL learning curves (if data provided) ────────────────────────────────
    if args.fl_rounds:
        fl_df = pd.read_csv(args.fl_rounds)
        for model_name in models:
            if "model" in fl_df.columns:
                subset = fl_df[fl_df["model"] == model_name]
            else:
                subset = fl_df
            if subset.empty:
                continue
            plot_fl_curves(
                {"rounds": subset["round"].tolist(),
                 "mae":    subset["mae"].tolist(),
                 "rmse":   subset["rmse"].tolist()},
                model_name
            )


if __name__ == "__main__":
    main()