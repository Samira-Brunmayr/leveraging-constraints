import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding adapted from
    https://github.com/huangnengCSU/SACall-basecaller/blob/master/transformer/modules.py"""

    def __init__(self, dim: int, dropout: float, max_len: int = 5000):
        super().__init__()

        # Compute positional embedding
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional embedding
        """
        x = x + nn.Parameter(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)
