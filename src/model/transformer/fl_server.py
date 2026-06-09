"""
fl_server.py  –  Transformer FL Server  (TempOut)

Pure FL — no warm start. Global model initialised from random weights.
Stronger proximal_mu to control client drift on non-IID station data.
"""
from __future__ import annotations
import logging
import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy.fedprox import FedProx
from train import load_config

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

config = load_config("config.yaml")
rounds = config["training"]["rounds"]


def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    result = {}
    for key in metrics[0][1].keys():
        result[key] = sum(n * m[key] for n, m in metrics) / total
    return result


logger.info(f"Transformer FL Server — {rounds} rounds, waiting for 8 clients")

fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=ServerConfig(num_rounds=rounds, round_timeout=3600),
    strategy=FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=0.3,    # stronger: limits client drift on non-IID stations
    ),
)