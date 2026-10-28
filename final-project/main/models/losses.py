"""Loss functions

Copied from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
"""

import torch
from torch import nn

CTC_BLANK = 0


def ctc_loss(y: torch.Tensor, p: torch.Tensor):
    """Calculates the ctc loss

    Args:
        y (tensor): tensor with labels [batch, len]
        p (tensor): tensor with predictions [len, batch, channels]

    Returns:
        loss (tensor): weighted sum of losses
    """
    ctc_loss = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True).to(y.device)
    y_len = torch.sum(y != CTC_BLANK, axis=1).to(y.device)
    p_len = torch.full((p.shape[1],), p.shape[0]).to(p.device)

    loss = ctc_loss(p, y, p_len, y_len)
    return loss
