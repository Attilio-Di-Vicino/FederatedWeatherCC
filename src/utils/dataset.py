"""
dataset.py
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset


class WeatherDataset(TorchDataset):
    """
    Sliding-window dataset for sequence-to-sequence weather prediction.

    Parameters
    ----------
    data : str | pd.DataFrame
        Path to a processed CSV or an already-loaded DataFrame.
    input_window : int
        Number of time-steps fed as encoder input.
    output_window : int
        Number of future time-steps to predict.
    feature_cols : list[str]
        Columns used as model *input* features (must NOT include target).
    target_cols : list[str]
        Column(s) to predict (e.g. ["RainRate"]).
    start_idx : int
        First row index to use (for train/val/test splitting).
    end_idx : int | None
        Last row index (exclusive). None → use all rows up to end.
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        input_window: int,
        output_window: int,
        feature_cols: list[str],
        target_cols: list[str],
        start_idx: int = 0,
        end_idx: int | None = None,
    ):
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        # Validate columns
        missing = [c for c in feature_cols + target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"WeatherDataset: missing columns {missing}")

        # Keep only the columns we need, in the right order
        df = df[feature_cols + target_cols].reset_index(drop=True)
        self.data = df.values  # numpy array

        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.n_features = len(feature_cols)
        self.n_targets = len(target_cols)
        self.input_window = input_window
        self.output_window = output_window

        if end_idx is None:
            end_idx = len(self.data)
        self.data = self.data[start_idx:end_idx]

    def __len__(self) -> int:
        return max(0, len(self.data) - self.input_window - self.output_window)

    def __getitem__(self, idx: int):
        # x: [input_window, n_features]
        x = self.data[idx: idx + self.input_window, : self.n_features]
        # y: [output_window, n_targets]
        y = self.data[
            idx + self.input_window: idx + self.input_window + self.output_window,
            self.n_features:,
        ]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )