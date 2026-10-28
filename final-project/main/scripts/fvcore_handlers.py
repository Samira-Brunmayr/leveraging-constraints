"""Handlers for fvcore"""

import functools
from collections import Counter

from fvcore.nn import FlopCountAnalysis


def scaled_dot_product_attention_flops(inputs, outputs):
    """Assume scaled dot product attention is softmax((Q @ K.T)/root(d_h)) @ V
    Inputs are size B x n_h x L x d_h
    For a single input and head we have Q = L x d_h, K.T = d_h x L, V = L x d_h

    Assuming fused multiplication addition = 1 FLOP:

    Q @ K.T is L * L * d_h FLOPs

    (Q @ K.T * 1/root(d_h)) is L * L * d_h + L*L FLOPs = L^2(d_h + 1) FLOPs

    For softmax assume exponentiaion is 20 FLOPs
    softmax(Q @ K.T/root(d_h)) is L^2(d_h + 1) + L^2 * 20 (exponentiation) + L^2 (divisions) + L^2 additions
    = L^2(d_h + 24)

    We ignore scaling and softmax FLOPs as these are negligible
    (Q @ K.T) @ V is L^2(d_h + 1 + 20 + 1 + 1) + L * L * d_h FLOPs
    = L^2(2*d_h + 24)

    So total number of FLOPs is L^2(2*d_h + 24)
    We then do this for every batch and every head to get:
    B * n_h * L^2(2*d_h + 24) FLOPs

    """
    Q = inputs[0]
    batch_size, n_heads, seq_len, d_head = tuple(Q.type().sizes())
    return Counter(
        {
            "scaled_dot_product_attention": batch_size
            * n_heads
            * seq_len
            * seq_len
            * (2 * d_head + 24)
        }
    )


def add_(inputs, outputs):
    """https://pytorch.org/docs/stable/generated/torch.add.html#torch.add
    Two tensor inputs followed by alpha (for scaling)"""
    sizes = inputs[0].type().sizes()
    # flops is just number of elements
    flops = functools.reduce(lambda x, y: x * y, sizes, 1)

    return Counter({"add_": flops})


def add(inputs, outputs):
    """https://pytorch.org/docs/stable/generated/torch.add.html#torch.add
    Two tensor inputs followed by alpha (for scaling)"""
    sizes = inputs[0].type().sizes()
    # flops is just number of elements
    flops = functools.reduce(lambda x, y: x * y, sizes, 1)
    return Counter({"add_": flops})


def div(inputs, outputs):
    """https://pytorch.org/docs/stable/generated/torch.div.html#torch.div
    Elementwise division, so one FLOP for each element in first arg"""
    sizes = inputs[0].type().sizes()
    # flops is just number of elements
    flops = functools.reduce(lambda x, y: x * y, sizes, 1)
    return Counter({"div": flops})


def unflatten(inputs, outputs):
    """Doesn't do any computations"""
    return Counter({"unflatten": 0})


def mul(inputs, outputs):
    """https://pytorch.org/docs/stable/generated/torch.mul.html
    Elementwise multiplication, so one FLOP for every element in first arg"""
    sizes = inputs[0].type().sizes()
    # flops is just number of elements
    flops = functools.reduce(lambda x, y: x * y, sizes, 1)
    return Counter({"mul": flops})


def log_softmax(inputs, outputs):
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
    Assume exponentiation and log are 20 FLOPs"""
    EXPONENTIATION_FLOPS = 20
    LOG_FLOPS = 20
    sizes = inputs[0].type().sizes()
    dim = sizes[inputs[1].toIValue()]
    number_elems = functools.reduce(lambda x, y: x * y, sizes, 1)

    exponentiation_flops_count = number_elems * EXPONENTIATION_FLOPS
    additions_flops_count = (number_elems // dim) * (dim - 1)
    log_flops_count = (number_elems // dim) * LOG_FLOPS
    division_flops = number_elems

    flops = (
        exponentiation_flops_count
        + additions_flops_count
        + log_flops_count
        + division_flops
    )

    return Counter({"log_softmax": flops})


def softmax(inputs, outputs):
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
    Assume exponentiation is 20 FLOPs"""
    EXPONENTIATION_FLOPS = 20
    sizes = inputs[0].type().sizes()
    dim = sizes[inputs[1].toIValue()]
    number_elems = functools.reduce(lambda x, y: x * y, sizes, 1)

    exponentiation_flops_count = number_elems * EXPONENTIATION_FLOPS
    additions_flops_count = (number_elems // dim) * (dim - 1)

    division_flops = number_elems

    flops = exponentiation_flops_count + additions_flops_count + division_flops

    return Counter({"softmax": flops})


def _record_function_enter_new(inputs, outputs):
    return Counter()


def _record_function_exit(inputs, outputs):
    return Counter()


HANDLERS = {
    "aten::scaled_dot_product_attention": scaled_dot_product_attention_flops,
    "aten::add_": add_,
    "aten::add": add,
    "aten::div": div,
    "aten::unflatten": unflatten,
    "aten::mul": mul,
    "aten::log_softmax": log_softmax,
    "aten::softmax": softmax,
    "profiler::_record_function_enter_new": _record_function_enter_new,
    "profiler::_record_function_exit": _record_function_exit,
}


def register(flop_counter: FlopCountAnalysis):
    """Register all handlers"""
    flop_counter.set_op_handle(**HANDLERS)
