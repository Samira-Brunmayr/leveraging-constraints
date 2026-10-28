"""old SACall model without LoRA implemented"""

from contextlib import nullcontext

import torch
from torch import nn

from main.models.basecaller import Basecaller

from .positional_encoding import PositionalEncoding


class SACall(Basecaller):
    """SACall encoder based on
    https://github.com/huangnengCSU/SACall-basecaller/blob/master/ctc/ctc_encoder.py
    """

    def __init__(
        self,
        device: torch.device,
        d_model: int,
        d_ff: int,
        n_head: int,
        num_encoder_layers: int,
        dropout=0.1,
        window_size: int = 2000,
    ):
        super().__init__(device=device)
        self.window_size = window_size
        self.kernel_size = 3
        self.dilation = 1
        self.padding = 1
        self.stride = 2
        self.total_stride = 4
        self.profiler_tags = [
            "convolutional encoder",
            "positional encoding",
            "transformer encoder",
            "fully connected decoder",
        ]

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=d_model // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.conv2 = nn.Conv1d(
            in_channels=d_model // 2,
            out_channels=d_model,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(d_model)
        self.convolutional_encoder = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.conv2, self.bn2, nn.ReLU()
        )
        self.position_encoding = PositionalEncoding(dim=d_model, dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="relu",
                    layer_norm_eps=1e-6,
                    batch_first=True,
                    norm_first=True,
                    bias=False,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.fc = nn.Linear(d_model, 5)

    def forward(self, signal, is_profiling: bool = False):
        """
        input: (B,1,C)
        output: (C/4,B,5)
        """
        with (
            torch.profiler.record_function("convolutional encoder")
            if is_profiling
            else nullcontext()
        ):
            h = self.convolutional_encoder(signal)
            h = h.transpose(1, 2)
        with torch.profiler.record_function("positional encoding"):
            x = self.position_encoding(h)
        for layer in self.encoder_layers:
            with (
                torch.profiler.record_function("transformer encoder")
                if is_profiling
                else nullcontext()
            ):
                x = layer(x)
        with (
            torch.profiler.record_function("fully connected decoder")
            if is_profiling
            else nullcontext()
        ):
            x = x.transpose(0, 1)
            x = self.ln(x)
            x = self.fc(x)
            return torch.nn.functional.log_softmax(x, dim=2)
