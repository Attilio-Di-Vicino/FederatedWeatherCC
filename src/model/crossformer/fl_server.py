"""
fl_server.py  (Crossformer)

Warm start: the global model is initialised from the centralised
trained checkpoint before FL rounds begin. This avoids clients
spending all their local epochs recovering from random initialisation.
"""
from __future__ import annotations
import logging
import os
import numpy as np
import torch
import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy.fedprox import FedProx
from train import load_config, build_model

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


# Load centralised checkpoint as initial global model weights
CKPT_PATH = "../../../data/trained_model/crossformer_tempout_final.pth"

def get_initial_parameters():
    if not os.path.exists(CKPT_PATH):
        logger.warning(
            f"Centralised checkpoint not found at {CKPT_PATH}. "
            "Starting from random weights. Train the centralised model first "
            "with: python train.py"
        )
        return None

    device = torch.device("cpu")
    model  = build_model(config, device)
    ckpt   = torch.load(CKPT_PATH, map_location="cpu")

    # Checkpoint may be a state dict directly or wrapped in a dict
    state_dict = ckpt if isinstance(ckpt, dict) and "model" not in ckpt else ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)

    params = [val.cpu().numpy() for val in model.state_dict().values()]
    logger.info(f"Warm start: loaded centralised weights from {CKPT_PATH}")
    return fl.common.ndarrays_to_parameters(params)


initial_parameters = get_initial_parameters()

logger.info(f"Crossformer FL Server — {rounds} rounds, waiting for 8 clients")

fl.server.start_server(
    server_address="0.0.0.0:8082",
    config=ServerConfig(num_rounds=rounds, round_timeout=3600),
    strategy=FedProx(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        proximal_mu=0.01,
    ),
)