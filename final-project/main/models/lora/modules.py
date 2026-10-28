import math
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LoRAOptions:
    q: bool
    k: bool
    v: bool
    attn_out: bool
    ffn: bool
    rank: int | None

    def use_lora(self) -> bool:
        options_set = any([self.q, self.k, self.v, self.attn_out, self.ffn])
        if options_set:
            if self.rank is not None:
                return True
            else:
                raise ValueError("Must specify lora rank with --lora-rank")
        else:
            return False


class LoRAMode(Enum):
    PRETRAIN = 0
    FINETUNE = 1


class LoRALinear(nn.Linear):
    """Implements a linear layer with low rank adaptation"""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        alpha=1.0,
        device=None,
        dtype=None,
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias, device, dtype)
        self.register_parameter("A", None)
        self.register_parameter("B", None)
        self.alpha = 1.0
        self.mode = None
        self.pretrain()
        self.rank = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.mode == LoRAMode.PRETRAIN:
            return super().forward(input)
        elif self.mode == LoRAMode.FINETUNE:
            return F.linear(
                input,
                self.weight + (self.alpha / self.rank) * (self.A @ self.B).T,
                self.bias,
            )

    def pretrain(self) -> None:
        if self.A is not None and self.B is not None:
            self.A.requires_grad = False
            self.B.requires_grad = False
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True
        self.mode = LoRAMode.PRETRAIN

    def finetune(self, rank: int | None, overwrite: bool = False) -> None:
        if self.rank is None:
            if rank is None:
                raise ValueError(
                    "Must provide rank when switching to finetune for the first time"
                )
            else:
                self.rank = rank
        elif self.rank != rank and rank is not None:
            if not overwrite:
                raise ValueError(
                    f"Overwriting current rank {self.rank} with rank {rank}, please set overwrite to True if this is intentional"
                )
            else:
                self.rank = rank
                self.register_parameter("A", None)
                self.register_parameter("B", None)
        if self.A is None and self.B is None:
            self.register_parameter(
                "A",
                nn.Parameter(
                    torch.randn((self.in_features, self.rank), **self.factory_kwargs)
                ),
            )
            self.register_parameter(
                "B",
                nn.Parameter(
                    torch.zeros((self.rank, self.out_features), **self.factory_kwargs)
                ),
            )
        self.A.requires_grad = True
        self.B.requires_grad = True
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.mode = LoRAMode.FINETUNE


class LoRAFFN(nn.Module):
    """Feedforward network adapted from
    https://github.com/huangnengCSU/SACall-basecaller/blob/master/transformer/modules.py
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            LoRALinear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            LoRALinear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x) + x  # residual connection
        return x

    def finetune(self, lora_options: LoRAOptions, overwrite: bool = False) -> None:
        if lora_options.ffn:
            self.ffn[1].finetune(rank=lora_options.rank, overwrite=overwrite)
            self.ffn[4].finetune(rank=lora_options.rank, overwrite=overwrite)


class LoRAMultiheadAttention(nn.Module):
    """MultiheadAttention adapted from
    https://github.com/huangnengCSU/SACall-basecaller/blob/master/transformer/modules.py
    """

    def __init__(self, n_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(LoRAMultiheadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = LoRALinear(d_model, d_model, bias=False)
        self.w_k = LoRALinear(d_model, d_model, bias=False)
        self.w_v = LoRALinear(d_model, d_model, bias=False)
        self.fc = LoRALinear(d_model, d_model, bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = (
            self.w_q(query).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
        )
        key = self.w_k(key).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = (
            self.w_v(value).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
        )

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -float("inf"))
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        output, self.attn = (p_attn @ value), p_attn

        # 3) "Concat" using a view and apply a final linear.
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.n_heads * self.d_k)
        )
        output = self.fc(output)

        return output, self.attn

    def finetune(self, lora_options: LoRAOptions, overwrite: bool = False) -> None:
        if lora_options.q:
            self.w_q.finetune(rank=lora_options.rank, overwrite=overwrite)
        if lora_options.k:
            self.w_k.finetune(rank=lora_options.rank, overwrite=overwrite)
        if lora_options.v:
            self.w_v.finetune(rank=lora_options.rank, overwrite=overwrite)
        if lora_options.attn_out:
            self.fc.finetune(rank=lora_options.rank, overwrite=overwrite)


class TransformerEncoderLayerLoRA(nn.Module):
    """transformer encoder layer adapted from
    https://github.com/huangnengCSU/SACall-basecaller/blob/master/transformer/modules.py
    """

    def __init__(self, d_model: int, d_ff: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = LoRAMultiheadAttention(
            n_heads=n_head, d_model=d_model, dropout=dropout
        )
        self.ffn = LoRAFFN(in_dim=d_model, hidden_dim=d_ff, dropout=dropout)
        self.ln = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, signal_emb: torch.Tensor, src_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_norm = self.ln(signal_emb)
        enc_out, enc_self_attn = self.attn(input_norm, input_norm, input_norm, src_mask)
        enc_out = signal_emb + self.dropout(enc_out)

        enc_out = self.ffn(enc_out)

        return enc_out, enc_self_attn

    def finetune(self, lora_options: LoRAOptions, overwrite: bool = False) -> None:
        self.attn.finetune(lora_options, overwrite)
        self.ffn.finetune(lora_options, overwrite)
