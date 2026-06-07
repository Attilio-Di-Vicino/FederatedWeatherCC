"""
fl_client.py  (Crossformer)
"""

from __future__ import annotations

import logging
import os
import sys

import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))
from dataset import WeatherDataset

from train import load_config, build_model, train_one_epoch, evaluate


def _make_loaders(config: dict, station_id: int):
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

    logger.info(f"[CLIENT {station_id}] {len(df)} rows")

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
        DataLoader(train_ds, shuffle=True,  **kwargs),  # shuffle for local training
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


class FLClient(fl.client.NumPyClient):
    def __init__(self, config: dict, station_id: int):
        self.config      = config
        self.station_id  = station_id
        self.device      = torch.device(
            config["training"]["device"] if torch.cuda.is_available() else "cpu"
        )
        self.target_cols   = config["data"]["target_cols"]
        self.local_epochs  = int(config["training"].get("local_epochs", 5))

        self.model = build_model(config, self.device)
        self.train_loader, self.val_loader, self.test_loader = \
            _make_loaders(config, station_id)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=1e-4,
        )

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(
            self.model.state_dict().keys(),
            [torch.tensor(p, device=self.device) for p in parameters],
        ))
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Multiple local epochs per round: critical for FL convergence
        for epoch in range(self.local_epochs):
            loss = train_one_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device,
            )
            logger.info(f"  [ws{self.station_id}] local epoch {epoch+1}/{self.local_epochs}"
                         f"  loss={loss:.5f}")

        val_results = evaluate(
            self.model, self.val_loader, self.device, self.target_cols,
        )
        metrics = {
            "val_mae":   float(val_results[0]["mae"]),
            "val_rmse":  float(val_results[0]["rmse"]),
            "val_skill": float(val_results[0]["skill"]),
        }
        logger.info(f"[ws{self.station_id}] fit done — "
                    f"val_mae={metrics['val_mae']:.4f}  "
                    f"val_skill={metrics['val_skill']:.4f}")

        os.makedirs("../../../data/trained_model", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"../../../data/trained_model/client_{self.station_id}_crossformer.pth",
        )
        return self.get_parameters(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        results = evaluate(
            self.model, self.test_loader, self.device, self.target_cols,
        )
        r = results[0]
        print(f"\n[Client {self.station_id}] CROSSFORMER – TempOut")
        print("=" * 55)
        print(f"  MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  Skill={r['skill']:.4f}")

        return (
            float(r["rmse"] ** 2),
            len(self.test_loader.dataset),
            {
                "mae":   float(r["mae"]),
                "rmse":  float(r["rmse"]),
                "skill": float(r["skill"]),
            },
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fl_client.py <station_id> <server_address>")
        sys.exit(1)

    station_id     = int(sys.argv[1])
    server_address = sys.argv[2]

    logger.info(f"Starting Crossformer FL Client  "
                f"station_id={station_id}  server={server_address}")

    cfg    = load_config("config.yaml")
    client = FLClient(cfg, station_id)
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
        transport="grpc-bidi",
    )