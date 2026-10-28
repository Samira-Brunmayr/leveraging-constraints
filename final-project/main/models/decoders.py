"""Decoding approaches"""

from typing import Any

import numpy as np
import torch
from fast_ctc_decode import beam_search, viterbi_search

BASES_CRF = "NACGT"


def decode_ctc_greedy(
    p: torch.Tensor,
    qstring: bool = False,
    qscale: float = 1.0,
    qbias: float = 1.0,
    collapse_repeats: bool = True,
    return_path: bool = False,
) -> list[str]:
    """Predict the bases in a greedy approach
    Args:
        p (tensor): [len, batch, classes]
        qstring (bool): whether to return the phredq scores
        qscale (float)
        qbias (float)
    Copied from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """

    alphabet = BASES_CRF
    decoded_predictions = list()
    p = p.exp().detach().cpu().numpy().astype(np.float32)
    for i in range(p.shape[1]):
        seq, path = viterbi_search(
            p[:, i, :],
            alphabet,
            qstring=qstring,
            qscale=qscale,
            qbias=qbias,
            collapse_repeats=collapse_repeats,
        )
        if return_path:
            decoded_predictions.append((seq, path))
        else:
            decoded_predictions.append(seq)

    return decoded_predictions


def decode_ctc_beamsearch(
    p: torch.Tensor,
    decoder_options: dict[str, Any] = dict(),
    beam_size: int = 3,
    beam_cut_threshold: float = 0.0,
    collapse_repeats: bool = True,
) -> list[str]:
    """Calls fast-ctc-decode beam search
    Adapted from https://github.com/marcpaga/basecalling_architectures/blob/main/src/classes.py
    """
    homopolymer_penalty = decoder_options.get("homopolymer_penalty", -1)
    max_repeats = decoder_options.get("max_repeats", -1)
    length = decoder_options.get("length", 0)
    length_lambda = decoder_options.get("length_lambda", -1)
    scale_by_timestep = decoder_options.get("scale_by_timestep", False)
    discard_by_expected_length = decoder_options.get("discard_by_expected_length", -1)
    kl_lambda = decoder_options.get("kl_lambda", 0)
    discard_by_kl = decoder_options.get("discard_by_kl", -1.0)
    entropy_ratio = decoder_options.get("entropy_ratio", 0.8)

    alphabet = BASES_CRF
    decoded_predictions = list()
    p = p.exp().detach().cpu().numpy().astype(np.float32)
    for i in range(p.shape[1]):
        seq, _ = beam_search(
            p[:, i, :],
            alphabet,
            beam_size=beam_size,
            beam_cut_threshold=beam_cut_threshold,
            collapse_repeats=collapse_repeats,
            homopolymer_penalty=homopolymer_penalty,
            max_repeats=max_repeats,
            length=length,
            length_lambda=length_lambda,
            scale_by_timestep=scale_by_timestep,
            discard_by_expected_length=discard_by_expected_length,
            kl_lambda=kl_lambda,
            discard_by_kl=discard_by_kl,
            entropy_ratio=entropy_ratio,
        )
        decoded_predictions.append(seq)

    return decoded_predictions
