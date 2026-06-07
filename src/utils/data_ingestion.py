"""
data_ingestion.py

Supports two raw dataset directory structures simultaneously:

  Structure A:  station/year/month/day/hourly_csv_files
  Structure B:  station/month/date_csv_files  (all year 2025)

Both are detected automatically per station directory.

Target  : TempOut
Features: station_id, Barometer, HumOut, RainDay,
          WindSpeed, WindSpeed10Min, wind_sin, wind_cos
Dropped : BarTrend (100% zero), SolarRad (79% NaN),
          UV (85% NaN), RainRate (99% zeros)
"""

from __future__ import annotations

import glob
import logging
import math
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)

# ── Column definitions ────────────────────────────────────────────────────────
RAW_KEEP_COLS = [
    "Barometer",
    "HumOut",
    "RainDay",
    "TempOut",
    "WindDir",
    "WindSpeed",
    "WindSpeed10Min",
]

VALIDITY_RANGES: dict[str, tuple[float, float]] = {
    "Barometer":      (800.0,  1100.0),
    "HumOut":         (0.0,    100.0),
    "RainDay":        (0.0,    500.0),
    "TempOut":        (-40.0,  60.0),
    "WindDir":        (0.0,    360.0),
    "WindSpeed":      (0.0,    100.0),
    "WindSpeed10Min": (0.0,    100.0),
}

FEATURE_COLS = [
    "station_id",
    "Barometer",
    "HumOut",
    "RainDay",
    "WindSpeed",
    "WindSpeed10Min",
    "wind_sin",
    "wind_cos",
]

TARGET_COL = "TempOut"

_FALLBACK = {
    "Barometer":      1013.0,
    "HumOut":         70.0,
    "RainDay":        0.0,
    "WindSpeed":      0.0,
    "WindSpeed10Min": 0.0,
    "wind_sin":       0.0,
    "wind_cos":       1.0,
}


# ── Station name → integer ID ─────────────────────────────────────────────────

def _station_name_to_id(name: str) -> int:
    parts = name.rstrip("/").split(".")
    for part in reversed(parts):
        if part.startswith("ws") and part[2:].isdigit():
            return int(part[2:])
    return abs(hash(name)) % 10_000


# ── Single-file loader (shared by both structures) ────────────────────────────

def _load_single_csv(filepath: Union[str, Path], station_id: int) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as exc:
        log.warning(f"Could not read {filepath}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Datetime — try all known column names
    datetime_col = next(
        (c for c in ("Datetime", "DatetimeWS", "timestamp") if c in df.columns),
        None,
    )
    if datetime_col is None:
        log.warning(f"No datetime column in {filepath}; skipping.")
        return pd.DataFrame()

    try:
        df["datetime"] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    except Exception:
        df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")

    df = df.dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame()

    # Select relevant columns
    keep = ["datetime"] + [c for c in RAW_KEEP_COLS if c in df.columns]
    df = df[keep].copy()
    for col in RAW_KEEP_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce to numeric and clamp to valid ranges
    for col in RAW_KEEP_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, (lo, hi) in VALIDITY_RANGES.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    df["station_id"] = int(station_id)
    return df


# ── Structure detection ───────────────────────────────────────────────────────

def _detect_structure(station_dir: Path) -> str:
    """
    Returns 'A' or 'B'.

    Structure A (old): station/year/month/day/*.csv
      — first-level subdirs are 4-digit years (e.g. 2026)

    Structure B (new): station/month/date.csv
      — first-level subdirs are 2-digit months (e.g. 01, 02, ..., 12)
    """
    subdirs = [d for d in station_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return "B"   # flat or no subdirs — try B
    first = subdirs[0].name
    if len(first) == 4 and first.isdigit():
        return "A"
    return "B"


def _collect_csvs_structure_a(station_dir: Path) -> list[Path]:
    """station/year/month/day/*.csv"""
    return sorted(station_dir.glob("*/*/*/*.csv"))


def _collect_csvs_structure_b(station_dir: Path) -> list[Path]:
    """station/month/date.csv"""
    return sorted(station_dir.glob("*/*.csv"))


# ── Per-station loader ────────────────────────────────────────────────────────

def load_station_data(station_dir: Union[str, Path]) -> pd.DataFrame:
    station_dir = Path(station_dir)
    station_id  = _station_name_to_id(station_dir.name)
    structure   = _detect_structure(station_dir)

    if structure == "A":
        csv_files = _collect_csvs_structure_a(station_dir)
    else:
        csv_files = _collect_csvs_structure_b(station_dir)

    if not csv_files:
        # Fallback: try the other structure
        csv_files = sorted(station_dir.rglob("*.csv"))

    if not csv_files:
        log.warning(f"No CSV files found under {station_dir}")
        return pd.DataFrame()

    log.info(f"[Station {station_id}] Structure {structure} — "
             f"{len(csv_files)} CSV files.")

    frames = [f for f in (_load_single_csv(fp, station_id) for fp in csv_files)
              if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)

    # Remove exact duplicate timestamps (can happen when files overlap)
    df = df.drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    log.info(f"[Station {station_id}] {len(df)} rows after dedup.")
    return df


def load_all_stations(root_dir: Union[str, Path]) -> pd.DataFrame:
    root_dir     = Path(root_dir)
    station_dirs = sorted(d for d in root_dir.iterdir()
                          if d.is_dir() and "meteo.ws" in d.name)
    if not station_dirs:
        raise FileNotFoundError(
            f"No station directories in {root_dir}. "
            "Expected names like 'it.uniparthenope.meteo.ws1'."
        )

    all_frames = [f for f in (load_station_data(sd) for sd in station_dirs)
                  if not f.empty]
    if not all_frames:
        raise ValueError("All station directories produced empty DataFrames.")

    merged = pd.concat(all_frames, ignore_index=True)
    log.info(f"Total rows loaded (raw, all stations): {len(merged)}")
    return merged


# ── Feature engineering ───────────────────────────────────────────────────────

def encode_wind(df: pd.DataFrame) -> pd.DataFrame:
    """Convert WindDir → sin/cos before resampling (averaging is correct on components)."""
    df = df.copy()
    wind_rad       = np.deg2rad(df["WindDir"].fillna(0.0))
    df["wind_sin"] = np.sin(wind_rad)
    df["wind_cos"] = np.cos(wind_rad)
    return df.drop(columns=["WindDir"])


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Average sub-hourly readings into 1-hour bins per station, then sort globally by time."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    hourly_groups = []
    for sid, grp in df.groupby("station_id", sort=True):
        grp       = grp.set_index("datetime").drop(columns=["station_id"])
        resampled = grp.resample("1h").mean()
        resampled["station_id"] = int(sid)
        hourly_groups.append(resampled.reset_index())

    hourly = pd.concat(hourly_groups, ignore_index=True)

    # Critical: sort by datetime globally so row-index splits == temporal splits
    hourly = hourly.sort_values("datetime").reset_index(drop=True)

    log.info(f"After resampling: {len(hourly)} hourly rows "
             f"(from {len(df)} raw rows).")
    return hourly


def impute_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill + interpolate on the hourly series, then median-impute residuals."""
    fill_cols = [
        "Barometer", "HumOut", "RainDay",
        "WindSpeed", "WindSpeed10Min",
        "wind_sin", "wind_cos",
        TARGET_COL,
    ]

    station_groups = []
    for sid, grp in df.groupby("station_id", sort=False):
        grp = grp.sort_values("datetime").copy()
        grp[fill_cols] = grp[fill_cols].ffill(limit=3)
        grp[fill_cols] = grp[fill_cols].interpolate(
            method="linear", limit=6, limit_direction="forward"
        )
        station_groups.append(grp)
    df = pd.concat(station_groups, ignore_index=True)

    # Drop rows where TARGET is still NaN
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    if before != len(df):
        log.info(f"Dropped {before - len(df)} hours with missing {TARGET_COL}.")

    # Median-impute remaining feature NaN
    feature_cols = [c for c in fill_cols if c != TARGET_COL]
    for col in feature_cols:
        n_nan = df[col].isnull().sum()
        if n_nan == 0:
            continue
        median_val = df[col].median()
        if math.isnan(median_val):
            median_val = _FALLBACK.get(col, 0.0)
            log.warning(f"  '{col}' entirely NaN → constant {median_val}")
        else:
            log.info(f"  Imputing {n_nan} NaN ({100*n_nan/len(df):.2f}%) "
                     f"in '{col}' with median={median_val:.4f}")
        df[col] = df[col].fillna(median_val)

    # Re-sort globally after per-station processing
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, cols: list[str]) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[cols].values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = scaler.transform(df[cols].values)
    return df


# ── NaN assertion ─────────────────────────────────────────────────────────────

def _assert_no_nan(df: pd.DataFrame):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    bad     = df[numeric].isnull().sum()
    bad     = bad[bad > 0]
    if not bad.empty:
        log.error("NaN still present:\n" + bad.to_string())
        raise ValueError("Dataset contains NaN after ingestion.")
    log.info("NaN assertion passed — dataset is clean.")


# ── End-to-end pipeline ───────────────────────────────────────────────────────

def build_processed_dataset(
    raw_root: Union[str, Path],
    output_csv: Union[str, Path],
    scale: bool = True,
) -> pd.DataFrame:
    df = load_all_stations(raw_root)
    df = encode_wind(df)
    df = resample_hourly(df)
    df = impute_hourly(df)

    all_cols = FEATURE_COLS + [TARGET_COL]
    all_cols = [c for c in all_cols if c in df.columns]
    df = df[["datetime"] + all_cols].copy()
    df["station_id"] = df["station_id"].astype(int)

    if scale:
        scale_cols = [c for c in all_cols if c != "station_id"]
        scaler = fit_scaler(df, scale_cols)
        df = apply_scaler(df, scaler, scale_cols)

    _assert_no_nan(df)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info(f"Saved → {output_csv}  ({len(df)} rows, {len(df.columns)} cols)")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Ingest raw weather CSVs → processed hourly dataset"
    )
    parser.add_argument(
        "raw_roots",
        nargs="+",
        help="One or more root directories containing station sub-directories. "
             "All are merged into a single output CSV.",
    )
    parser.add_argument("--output", default="data/processed_data/dataset.csv")
    parser.add_argument("--no-scale", action="store_true")
    args = parser.parse_args()

    if len(args.raw_roots) == 1:
        build_processed_dataset(args.raw_roots[0], args.output, scale=not args.no_scale)
    else:
        # Multiple roots: load each, concatenate, then process together
        import tempfile, os
        frames = []
        for root in args.raw_roots:
            log.info(f"Loading from {root}")
            frames.append(load_all_stations(root))
        merged = pd.concat(frames, ignore_index=True)
        log.info(f"Combined raw rows: {len(merged)}")

        merged = encode_wind(merged)
        merged = resample_hourly(merged)
        merged = impute_hourly(merged)

        all_cols = FEATURE_COLS + [TARGET_COL]
        all_cols = [c for c in all_cols if c in merged.columns]
        merged = merged[["datetime"] + all_cols].copy()
        merged["station_id"] = merged["station_id"].astype(int)

        if not args.no_scale:
            scale_cols = [c for c in all_cols if c != "station_id"]
            scaler = fit_scaler(merged, scale_cols)
            merged = apply_scaler(merged, scaler, scale_cols)

        _assert_no_nan(merged)

        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_csv, index=False)
        log.info(f"Saved → {output_csv}  ({len(merged)} rows)")