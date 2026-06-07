"""
model.py (Crossformer)

No target-specific logic lives here; the architecture is generic.
input_dim / output_dim are driven entirely by config.yaml.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Crossformer(nn.Module):
    """
    Crossformer: a Transformer that processes the input sequence in
    non-overlapping blocks (local attention) and then applies a second
    Transformer over the block representations (global attention).

    Parameters
    ----------
    input_dim : int
        Number of input features per time-step.
    output_dim : int
        Number of output targets per predicted time-step.
    d_model : int
        Internal embedding dimension.
    block_size : int
        Number of time-steps per local block.
        input_window must be divisible by block_size.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of TransformerEncoder layers used for both local and global attention.
    dropout : float
        Dropout probability.
    output_window : int
        Number of future time-steps to predict.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        block_size: int = 4,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_window: int = 4,
    ):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.output_window = output_window

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Aggregate each block to a single vector
        self.aggregator = nn.AdaptiveAvgPool1d(1)

        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape [B, T, F]

        Returns
        -------
        Tensor of shape [B, output_window, output_dim]
        """
        B, T, F = x.shape
        if T % self.block_size != 0:
            raise ValueError(
                f"Crossformer: input_window ({T}) must be divisible by "
                f"block_size ({self.block_size})."
            )

        x = self.input_proj(x)           # [B, T, d_model]
        x = self.pos_encoder(x)

        n_blocks = T // self.block_size

        # Local attention within each block
        blocks = x.view(B, n_blocks, self.block_size, self.d_model)
        blocks = blocks.reshape(-1, self.block_size, self.d_model)       # [B*n, bs, d]
        encoded_blocks = self.encoder(blocks)                            # [B*n, bs, d]

        # Aggregate each block → one vector
        agg = self.aggregator(encoded_blocks.transpose(1, 2)).squeeze(-1)  # [B*n, d]
        agg = agg.view(B, n_blocks, self.d_model)                          # [B, n, d]

        # Global attention across blocks
        global_encoded = self.encoder(agg)                                 # [B, n, d]

        out = self.output_proj(global_encoded)                             # [B, n, output_dim]
        out = out[:, -self.output_window:, :]                              # [B, ow, output_dim]
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)