"""
fl_server.py  –  Crossformer FL Server  (TempOut)
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
rounds      = config["training"]["rounds"]
proximal_mu = float(config["training"].get("proximal_mu", 0.3))


def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    result = {}
    for key in metrics[0][1].keys():
        result[key] = sum(n * m[key] for n, m in metrics) / total
    return result


logger.info(
    f"Crossformer FL Server — {rounds} rounds  "
    f"proximal_mu={proximal_mu}  waiting for 8 clients"
)

fl.server.start_server(
    server_address="0.0.0.0:8082",
    config=ServerConfig(num_rounds=rounds, round_timeout=3600),
    strategy=FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=0.1,
    ),
)