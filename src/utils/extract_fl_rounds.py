"""
extract_fl_rounds.py

Parses the Flower server log output and extracts per-round aggregated
evaluate metrics (MAE, RMSE) into a CSV file ready for plot_results.py.

Two usage modes:

  MODE 1 — parse a saved log file:
    python src/utils/extract_fl_rounds.py --log logs/fl_server_crossformer.out \
        --output fl_rounds_crossformer.csv

  MODE 2 — pipe server output directly:
    python fl_server.py 2>&1 | tee logs/server.log | \
        python src/utils/extract_fl_rounds.py --stdin --output fl_rounds.csv

  MODE 3 — interactive: paste server output, then Ctrl+D:
    python src/utils/extract_fl_rounds.py --output fl_rounds.csv

The script looks for lines produced by the weighted_average aggregation,
which Flower prints in the format:

    [INFO] evaluate_metrics_aggregation_fn: {'mae': 0.123, 'rmse': 0.456, ...}

or the round summary lines:

    [INFO] fit progress: (round, loss, metrics, timedelta)

It handles both formats automatically.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


# ── Regex patterns ────────────────────────────────────────────────────────────

# Flower prints aggregated evaluate metrics like:
#   evaluate_metrics_aggregation_fn: {'mae': 0.12, 'rmse': 0.34}
RE_EVAL_METRICS = re.compile(
    r"evaluate_metrics_aggregation_fn.*?'mae':\s*([\d.]+).*?'rmse':\s*([\d.]+)",
    re.IGNORECASE,
)

# Flower also prints fit progress per round:
#   fit progress: (1, 0.045, {'val_mae': 0.12, 'val_rmse': 0.34}, datetime)
RE_FIT_PROGRESS = re.compile(
    r"fit progress.*?\((\d+),\s*([\d.]+),\s*\{(.*?)\}",
    re.IGNORECASE,
)

# Round number from lines like:
#   [INFO] configure_evaluate: strategy sampled ... (round 3)
# or
#   [INFO] Starting round 5
RE_ROUND = re.compile(r"(?:round|starting round)\s+(\d+)", re.IGNORECASE)

# Aggregated metrics block — Flower sometimes prints:
#   [INFO] Results: [(ClientProxy, EvaluateRes(...))]
# We look for the aggregated line instead:
#   aggregated_evaluate_metrics: {'mae': ..., 'rmse': ...}
RE_AGG = re.compile(
    r"aggregated.*?'mae':\s*([\d.eE+\-]+).*?'rmse':\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)


def parse_log(lines: list[str]) -> list[dict]:
    """
    Parse log lines and return a list of dicts with keys:
        round, mae, rmse
    """
    results: dict[int, dict] = {}
    current_round = None

    for line in lines:
        # Track current round number
        m_round = RE_ROUND.search(line)
        if m_round:
            current_round = int(m_round.group(1))

        # Try evaluate metrics pattern
        m_eval = RE_EVAL_METRICS.search(line)
        if m_eval and current_round is not None:
            mae  = float(m_eval.group(1))
            rmse = float(m_eval.group(2))
            results[current_round] = {"round": current_round, "mae": mae, "rmse": rmse}
            continue

        # Try aggregated metrics pattern
        m_agg = RE_AGG.search(line)
        if m_agg and current_round is not None:
            mae  = float(m_agg.group(1))
            rmse = float(m_agg.group(2))
            results[current_round] = {"round": current_round, "mae": mae, "rmse": rmse}
            continue

        # Try fit progress pattern (backup)
        m_fit = RE_FIT_PROGRESS.search(line)
        if m_fit:
            rnd     = int(m_fit.group(1))
            metrics = m_fit.group(3)
            mae_m  = re.search(r"'(?:val_)?mae':\s*([\d.eE+\-]+)", metrics)
            rmse_m = re.search(r"'(?:val_)?rmse':\s*([\d.eE+\-]+)", metrics)
            if mae_m and rmse_m:
                results[rnd] = {
                    "round": rnd,
                    "mae":   float(mae_m.group(1)),
                    "rmse":  float(rmse_m.group(1)),
                }

    return sorted(results.values(), key=lambda x: x["round"])


def write_csv(rows: list[dict], output_path: str):
    if not rows:
        print("WARNING: No round metrics found in the input. "
              "Check that your log contains evaluate metric lines.")
        print("\nExpected format in server log:")
        print("  evaluate_metrics_aggregation_fn: {'mae': 0.123, 'rmse': 0.456}")
        print("\nFalling back to manual entry mode.")
        rows = manual_entry()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rounds → {out}")
    print("\nPreview:")
    print(f"{'Round':>7}  {'MAE':>10}  {'RMSE':>10}")
    print("-" * 32)
    for r in rows:
        print(f"{r['round']:>7}  {r['mae']:>10.6f}  {r['rmse']:>10.6f}")


def manual_entry() -> list[dict]:
    """Prompt the user to enter round metrics manually."""
    print("\nManual entry mode.")
    print("Enter: round,mae,rmse  (one per line, empty line to finish)")
    print("Example:  1,0.18,0.24")
    rows = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        parts = line.split(",")
        if len(parts) != 3:
            print("  Format: round,mae,rmse")
            continue
        try:
            rows.append({
                "round": int(parts[0]),
                "mae":   float(parts[1]),
                "rmse":  float(parts[2]),
            })
        except ValueError:
            print("  Could not parse — try again.")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Extract FL round metrics from Flower server log → CSV"
    )
    parser.add_argument(
        "--log", default=None,
        help="Path to saved server log file (e.g. logs/server.log)"
    )
    parser.add_argument(
        "--stdin", action="store_true",
        help="Read log from stdin (for piping server output)"
    )
    parser.add_argument(
        "--output", default="fl_rounds.csv",
        help="Output CSV path (default: fl_rounds.csv)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Optional: add a 'model' column with this value (e.g. crossformer)"
    )
    args = parser.parse_args()

    if args.stdin:
        print("Reading from stdin... (Ctrl+D to finish)")
        lines = sys.stdin.readlines()
    elif args.log:
        lines = Path(args.log).read_text().splitlines()
    else:
        print("No log source specified. Entering manual mode.")
        print("(Use --log <file> or --stdin to parse automatically)\n")
        rows = manual_entry()
        write_csv(rows, args.output)
        return

    rows = parse_log(lines)

    if args.model:
        for r in rows:
            r["model"] = args.model

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()