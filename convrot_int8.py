"""ConvRot W8A8 quantization of the frozen LoRA base — int8 tensor cores in the training forward.

Implements the int8 (W8A8) variant of *"ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for
Diffusion Transformers"* (arXiv:2512.03673). Self-contained on torch: no torchao, no Triton.

**Why the rotation.** Weights and activations are rotated by a block *regular* Hadamard transform —
Kronecker powers of the 4x4 regular Hadamard, orthonormal and symmetric, hence its own inverse. The
standard Hadamard has an all-ones row that concentrates a block's mean into one coordinate; the
regular one has constant row sums instead, so it smooths outliers along rows and columns
symmetrically. That is what makes a single scale per row safe: outliers are spread rather than
clipped, which is the failure mode coarse per-row scaling otherwise runs into.

The rotation costs nothing in accuracy: ``R`` is folded into the weight once, offline, and applied to
the activation at runtime, so it cancels inside the matmul (``x R (W R)^T = x W^T``). It is
deterministic — a fixed matrix, no randomness — and quantization is plain rounding.

**The training path is quantized too, not only inference.** The forward runs
``torch._int_mm`` on int8 tensor cores, available from Ampere onward. The backward is a
straight-through estimator: the gradient is taken against the dequantized weight, and only the input
gradient is produced. There is no weight gradient, so this composes with a frozen base carrying
adapters and does **not** compose with a full fine-tune, where the weights are what is being learned.

Call ``quantize_dit_int8`` BEFORE ``inject_lora`` so the adapters wrap quantized bases and stay bf16.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ACT_QMAX = 127
_hadamard_cache: dict = {}


def largest_pow4_divisor(d: int) -> int:
    """Largest power of four dividing ``d`` — the biggest usable rotation block."""
    h = 1
    while d % (h * 4) == 0:
        h *= 4
    return h


def regular_hadamard(rot_size: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """Kronecker powers of the 4x4 regular Hadamard, orthonormal. Symmetric, so self-inverse."""
    key = (rot_size, str(device), dtype)
    if key not in _hadamard_cache:
        r4 = torch.tensor([[1.0, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                          dtype=torch.float64)
        h = r4.clone()
        while h.shape[0] < rot_size:
            h = torch.kron(h, r4)
        if h.shape[0] != rot_size:
            raise ValueError(f"rot_size {rot_size} is not a power of 4")
        _hadamard_cache[key] = (h / rot_size ** 0.5).to(device=device, dtype=dtype)
    return _hadamard_cache[key]


def rotate(x: torch.Tensor, rot_size: int) -> torch.Tensor:
    """Apply the block rotation along the last dim. Self-inverse, so it cancels in the matmul."""
    if rot_size == 1:
        return x
    h = regular_hadamard(rot_size, x.device, x.dtype)
    shape = x.shape
    return torch.matmul(x.reshape(-1, shape[-1] // rot_size, rot_size), h).reshape(shape)


def quantize_int8_rows(x: torch.Tensor, qmax: int = ACT_QMAX):
    """Symmetric per-row quantization to [-qmax, qmax]. Returns (int8 (rows, K), fp32 scales)."""
    xf = x.float()
    scales = xf.abs().amax(dim=1) / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    q = torch.round(xf / scales.unsqueeze(1)).clamp_(-qmax, qmax).to(torch.int8)
    return q, scales


def _act_quant_padded(x: torch.Tensor, qmax: int):
    """Per-token activation quantization, rows padded to a multiple of 32 for ``torch._int_mm``."""
    q, scales = quantize_int8_rows(x, qmax)
    rows = q.shape[0]
    rows_pad = -(-rows // 32) * 32
    if rows_pad != rows:
        q = F.pad(q, (0, 0, 0, rows_pad - rows))
        scales = F.pad(scales, (0, rows_pad - rows), value=1.0)
    return q, scales


def _epilogue(i32, a_scales, w_scales, bias, out_dtype: torch.dtype) -> torch.Tensor:
    """``out = i32 * a_scales[:, None] * w_scales[None, :] (+ bias)``, accumulated in fp32."""
    out = i32.float() * w_scales
    out = out * a_scales.unsqueeze(1)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


# The forward VALUE is the real int8 tensor-core gemm; the gradient is the straight-through estimate
# dy/dx = dequant(W_rot). Registered as a custom op so it survives torch.compile, and the backward
# re-dequantizes from int8 rather than saving a bf16 copy — ``x`` is never saved, so this holds less
# memory than ``F.linear``.
@torch.library.custom_op("krea2::convrot_int8_linear", mutates_args=())
def convrot_int8_linear(x2d: torch.Tensor, qdata: torch.Tensor, w_scales_u8: torch.Tensor,
                        bias: Optional[torch.Tensor], act_qmax: int,
                        out_dtype: str) -> torch.Tensor:
    m = x2d.shape[0]
    aq, a_s = _act_quant_padded(x2d, act_qmax)
    i32 = torch._int_mm(aq, qdata.t())
    return _epilogue(i32[:m], a_s[:m], w_scales_u8.view(torch.float32), bias,
                     getattr(torch, out_dtype))


@convrot_int8_linear.register_fake
def _convrot_int8_linear_fake(x2d, qdata, w_scales_u8, bias, act_qmax, out_dtype):
    return torch.empty(x2d.shape[0], qdata.shape[0], device=x2d.device,
                       dtype=getattr(torch, out_dtype))


def _setup_context(ctx, inputs, output):
    _, qdata, w_scales_u8, _, _, _ = inputs
    ctx.save_for_backward(qdata, w_scales_u8)


def _backward(ctx, grad):
    qdata, w_scales_u8 = ctx.saved_tensors
    w_scales = w_scales_u8.view(torch.float32).to(grad.dtype)
    w = qdata.to(grad.dtype) * w_scales.unsqueeze(1)
    return grad @ w, None, None, None, None, None


convrot_int8_linear.register_autograd(_backward, setup_context=_setup_context)


class ConvRotInt8Linear(nn.Module):
    """Frozen ``nn.Linear`` stored as rotated int8 weights with a per-output-row scale.

    Exposes ``in_features`` / ``out_features`` / ``weight`` so it drops in as a ``LoRALinear`` base,
    matching :class:`quantize.Fp8Linear`.
    """

    is_quant_linear = True

    def __init__(self, lin: nn.Linear, rot_size: int = 256):
        super().__init__()
        rot = min(rot_size, largest_pow4_divisor(lin.in_features))
        q, scales = quantize_int8_rows(rotate(lin.weight.data.float(), rot))
        self.register_buffer("qdata", q, persistent=False)
        # fp32 scales kept as a uint8 byte view: a later ``module.to(dtype=...)`` would otherwise
        # cast them and silently change the dequantization.
        self.register_buffer("scales_u8", scales.view(torch.uint8), persistent=False)
        self.bias = (nn.Parameter(lin.bias.data.clone(), requires_grad=False)
                     if lin.bias is not None else None)
        self.in_features, self.out_features, self.rot_size = lin.in_features, lin.out_features, rot

    @property
    def weight(self) -> torch.Tensor:  # device/dtype probe for LoRALinear
        return self.qdata

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xr = rotate(x.reshape(-1, shape[-1]), self.rot_size)
        y = convrot_int8_linear(xr, self.qdata, self.scales_u8, self.bias, ACT_QMAX,
                                str(x.dtype).split(".")[-1])
        return y.reshape(*shape[:-1], self.out_features)


def eligible(lin: nn.Linear, rot_size: int = 256) -> bool:
    """``in_features`` divisible by 16, ``out_features`` by 8, rotation block at least 16."""
    return (lin.in_features % 16 == 0 and lin.out_features % 8 == 0
            and min(rot_size, largest_pow4_divisor(lin.in_features)) >= 16)


def int8_gemm_supported(device=None) -> bool:
    """``torch._int_mm`` needs int8 tensor cores — Ampere (sm_80) and newer."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def quantize_dit_int8(dit, rot_size: int = 256) -> int:
    """Replace the transformer blocks' ``nn.Linear`` with :class:`ConvRotInt8Linear` in place.

    Leaves ``first`` / ``last`` / text-fusion / time-MLP alone, as ``quantize_dit_fp8`` does: they are
    small and sensitive. Returns the number of layers quantized. Call BEFORE ``inject_lora``.
    """
    targets = []
    for block in dit.blocks:
        for mod in block.modules():
            for name, child in mod.named_children():
                if isinstance(child, nn.Linear) and eligible(child, rot_size):
                    targets.append((mod, name, child))
    for mod, name, child in targets:
        setattr(mod, name, ConvRotInt8Linear(child, rot_size))
    return len(targets)
