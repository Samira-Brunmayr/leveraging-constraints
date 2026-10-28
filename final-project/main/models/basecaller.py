"""Basecaller abstract class"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class Basecaller(ABC, nn.Module):
    def __init__(
        self,
        device: torch.device,
    ):
        super().__init__()

        self.device = device
        self.init_weights()

    @abstractmethod
    def forward(self, batch: torch.Tensor, is_profiling: bool = False):
        """Forward through the network"""
        raise NotImplementedError()

    def init_weights(self):
        """Initialize weights from uniform distribution"""
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
