"""SACall model that can be finetuned with LoRA"""

from contextlib import nullcontext

import torch
from torch import nn

from main.models.basecaller import Basecaller
from main.models.lora.modules import LoRAOptions, TransformerEncoderLayerLoRA
from main.models.sacall.positional_encoding import PositionalEncoding


class SACallLoRA(Basecaller):
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
        self.d_model = d_model
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
                TransformerEncoderLayerLoRA(
                    d_model=d_model, d_ff=d_ff, n_head=n_head, dropout=dropout
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.fc = nn.Linear(d_model, 5)

    def forward(self, signal, is_profiling: bool = False):
        """
        signal is a tensor shape of (B,1,C)
        output is a tensor of (C/4,B,5)
        """
        batch_size = signal.size(0)
        chunk_size = signal.size(2)
        with (
            torch.profiler.record_function("convolutional encoder")
            if is_profiling
            else nullcontext()
        ):
            # (B,1,C)
            h = self.convolutional_encoder(signal)
            # (B,d_model,C/4)
            h = h.transpose(1, 2)
            # (B,C/4,d_model)
        with torch.profiler.record_function("positional encoding"):
            x = self.position_encoding(h)
        src_mask = torch.zeros(
            batch_size, 1, chunk_size // self.total_stride, dtype=bool
        ).to(self.device)
        for layer in self.encoder_layers:
            with (
                torch.profiler.record_function("transformer encoder")
                if is_profiling
                else nullcontext()
            ):
                x, _ = layer(x, src_mask)
        # (B, C/4, d_model)
        with (
            torch.profiler.record_function("fully connected decoder")
            if is_profiling
            else nullcontext()
        ):
            # (B, C/4, d_model)
            x = x.transpose(0, 1)
            # (C/4, B, d_model)
            x = self.ln(x)
            x = self.fc(x)
            # (C/4, B, 5)
            return torch.nn.functional.log_softmax(x, dim=2)

    def finetune(self, lora_options: LoRAOptions, overwrite: bool = False) -> None:
        """Set the LoRA layers to finetune"""
        for layer in self.encoder_layers:
            layer.finetune(lora_options, overwrite)

    def bitfit(self, train=True) -> None:
        """Unfreeze all bias terms within each transformer encoder layer"""
        for layer in self.encoder_layers:
            if train:
                # unfreeze layer norm bias
                if layer.ln.bias is not None:
                    layer.ln.bias.requires_grad = True

                # unfreeze feed forward network biases
                # unfreeze the layer norm
                if layer.ffn.ffn[0].bias is not None:
                    layer.ffn.ffn[0].bias.requires_grad = True

                # unfreeze the linear layer biases
                ffn_layers = [layer.ffn.ffn[1], layer.ffn.ffn[4]]
                for ffn_layer in ffn_layers:
                    if ffn_layer.bias is not None:
                        ffn_layer.bias.requires_grad = True

            # unfreeze attention weights
            attention_layers = [
                layer.attn.w_q,
                layer.attn.w_k,
                layer.attn.w_v,
                layer.attn.fc,
            ]
            for attn_layer in attention_layers:
                # first create biases
                if attn_layer.bias is None:
                    attn_layer.bias = nn.Parameter(torch.zeros(attn_layer.out_features))

            if train:
                attn_layer.bias.requires_grad = True

    def decoder_finetune(self) -> None:
        for param in self.fc.parameters():
            param.requires_grad = True
