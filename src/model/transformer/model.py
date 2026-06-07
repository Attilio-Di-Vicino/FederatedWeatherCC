"""
model.py (Transformer)

No target-specific logic lives here; the architecture is generic.
input_dim / output_dim are driven entirely by config.yaml.  The decoder 
receives a zero-initialised tensor of shape [B, output_window, input_dim] 
as its target sequence – a standard approach for multi-step forecasting 
with a Transformer decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class myTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for multi-step time-series forecasting.

    Parameters
    ----------
    input_dim : int
        Number of input features (encoder + decoder projection).
    output_dim : int
        Number of output targets per time-step.
    d_model : int
        Internal embedding dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of encoder AND decoder layers.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_linear  = nn.Linear(input_dim, d_model)
        self.pos_encoder   = PositionalEncoding(d_model, dropout)
        self.transformer   = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.output_linear = nn.Linear(d_model, output_dim)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        source : [B, T_in,  input_dim]
        target : [B, T_out, input_dim]  (zero tensor during inference)

        Returns
        -------
        [B, T_out, output_dim]
        """
        source = self.pos_encoder(self.input_linear(source))
        target = self.pos_encoder(self.input_linear(target))
        out    = self.transformer(source, target)
        return self.output_linear(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)