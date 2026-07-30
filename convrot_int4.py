"""ConvRot 4-bit quantization of the frozen LoRA base — the lowest-VRAM option here.

The 4-bit setting of *"ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion
Transformers"* (arXiv:2512.03673), of which :mod:`convrot_int8` implements the 8-bit one. The
regular-Hadamard rotation and the activation-side machinery are imported from there rather than
restated; what is new here is the storage and the two decisions below, both of which were made from
measurement rather than from the shape of the 8-bit code.

**Group scales, not per-row.** The 8-bit path scales per output row. Repeating that at four bits does
not work: a per-row 4-bit grid lands at a large fraction of the weight's own norm in error, against
roughly one percent for 8-bit. Scaling per group of ``group_size`` inputs is what makes four bits
usable at all. The rotation still earns its place on top of grouping -- it is what stops a single
outlier from setting the scale for its whole group.

**Asymmetric, anchored on the group's own minimum.** Every level is spent where the group's values
actually are, rather than on a symmetric range half of which may be empty. Measured on this model's own weights, that beats
both a symmetric grid and the normal-float (NF4) codebook that 4-bit tooling generally reaches for --
NF4's levels are placed for a standard normal and cluster near zero, which spends resolution where a
rotated group does not need it. The ordering is worth stating because it inverts on synthetic
heavy-tailed matrices, where a symmetric grid comes out ahead -- so a codebook chosen for this model
has to be compared on this model's weights, not on a plausible stand-in for them.

**This is a memory lever only.** Unlike ``int8``, which runs its matmul on int8 tensor cores, four
bits has no GEMM on any hardware torch exposes, and per-group scales cannot be folded into a single
fused epilogue anyway. The weight is therefore unpacked and dequantized one layer at a time and the
matmul runs in the activation's own dtype: compute does not improve and no int8 tensor cores are
required. Activations are not quantized, so unlike the 8-bit path this adds no activation error.

The packed codes are a quarter of bf16, but the per-group scale and offset are stored alongside them,
so the realised saving is somewhat less than 4x -- at the default group size they cost an extra bit
per weight. Halving the group would buy about a further point of accuracy and cost another bit;
doubling it does the reverse. Sixty-four is the middle of that trade, not a hardware constraint.

The backward is the same straight-through estimate as the int8 path -- the gradient is taken against
the dequantized weight and only the input gradient is produced. Keeping it inside a custom op is what
stops autograd from saving the dequantized bf16 weight of *every* layer until backward, which would
undo the entire saving.

Call ``quantize_dit_int4`` BEFORE ``inject_lora`` so the adapters wrap quantized bases and stay bf16.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from convrot_int8 import largest_pow4_divisor, rotate

QMAX = 15                 # 4-bit codes are unsigned 0..15; the per-group minimum carries the offset
GROUP_SIZE = 64           # inputs per scale
META_DTYPE = torch.bfloat16   # storage dtype of the per-group scale and offset


def quantize_int4_groups(x: torch.Tensor, group_size: int = GROUP_SIZE,
                         meta_dtype: torch.dtype = torch.bfloat16):
    """Asymmetric 4-bit quantization with a scale and an offset per ``group_size`` inputs.

    Returns (uint8 codes ``(rows, K)`` in ``[0, 15]``, scales, group minima, the latter two
    ``(rows, K // group_size)`` in ``meta_dtype``). Reconstruction is ``code * scale + minimum``, so
    all sixteen levels land inside the range the group actually occupies.

    The offset is stored as the group's MINIMUM rather than as an integer zero point. The integer
    form (``code - zero``) has to clamp that zero into ``[0, 15]``, which silently collapses any
    group that does not straddle zero -- an all-positive group saturates every code at 15 -- and
    turns a constant group into a division by a placeholder scale. Storing the minimum directly is
    the same grid without either failure, and it makes a constant group exact.

    The scale and offset are ROUNDED TO ``meta_dtype`` BEFORE the codes are computed, so the codes
    are chosen against the values dequantization will actually use. Rounding them afterwards would
    let the two disagree and add a second error on top of the grid's own.
    """
    rows, k = x.shape
    if k % group_size:
        raise ValueError(f"in_features {k} is not divisible by group_size {group_size}")
    xg = x.float().reshape(rows, k // group_size, group_size)
    lo, hi = xg.amin(dim=-1), xg.amax(dim=-1)
    scales = ((hi - lo) / QMAX).clamp_min(1e-12)      # constant group -> code 0 and an exact minimum
    scales, lo = scales.to(meta_dtype), lo.to(meta_dtype)
    q = torch.round((xg - lo.float().unsqueeze(-1))
                    / scales.float().unsqueeze(-1)).clamp_(0, QMAX).to(torch.uint8)
    return q.reshape(rows, k), scales, lo


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack uint8 codes in ``[0, 15]`` two per byte: column ``2i`` low nibble, ``2i+1`` high."""
    if q.shape[-1] % 2:
        raise ValueError(f"packing needs an even number of columns, got {q.shape[-1]}")
    lo = q[:, 0::2].to(torch.uint8) & 0xF
    hi = q[:, 1::2].to(torch.uint8) & 0xF
    return lo | (hi << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4` -> uint8 codes in ``[0, 15]``."""
    lo = packed & 0xF
    hi = packed.bitwise_right_shift(4) & 0xF
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, mins: torch.Tensor,
                    group_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Rebuild the rotated weight: ``code * scale + minimum``, per group."""
    q = unpack_int4(packed)
    rows, k = q.shape
    qg = q.reshape(rows, k // group_size, group_size).to(dtype)
    return (qg * scales.to(dtype).unsqueeze(-1) + mins.to(dtype).unsqueeze(-1)).reshape(rows, k)


# The dequantized weight lives only for the duration of one layer's matmul. Wrapping this in a custom
# op (rather than dequantizing in `forward` and calling F.linear) is what keeps it that way: plain
# autograd would save the bf16 weight of every quantized layer until backward. It also avoids saving
# ``x``. Registered as a custom op so it survives torch.compile.
@torch.library.custom_op("krea2::convrot_int4_linear", mutates_args=())
def convrot_int4_linear(x2d: torch.Tensor, packed: torch.Tensor, scales_u8: torch.Tensor,
                        mins_u8: torch.Tensor, bias: Optional[torch.Tensor],
                        group_size: int) -> torch.Tensor:
    w = dequantize_int4(packed, scales_u8.view(META_DTYPE), mins_u8.view(META_DTYPE),
                        group_size, x2d.dtype)
    return F.linear(x2d, w, bias.to(x2d.dtype) if bias is not None else None)


@convrot_int4_linear.register_fake
def _convrot_int4_linear_fake(x2d, packed, scales_u8, mins_u8, bias, group_size):
    return torch.empty(x2d.shape[0], packed.shape[0], device=x2d.device, dtype=x2d.dtype)


def _setup_context(ctx, inputs, output):
    _, packed, scales_u8, mins_u8, _, group_size = inputs
    ctx.save_for_backward(packed, scales_u8, mins_u8)
    ctx.group_size = group_size


def _backward(ctx, grad):
    packed, scales_u8, mins_u8 = ctx.saved_tensors
    w = dequantize_int4(packed, scales_u8.view(META_DTYPE), mins_u8.view(META_DTYPE),
                        ctx.group_size, grad.dtype)
    return grad @ w, None, None, None, None, None


convrot_int4_linear.register_autograd(_backward, setup_context=_setup_context)


class ConvRotInt4Linear(nn.Module):
    """Frozen ``nn.Linear`` stored as rotated 4-bit weights, two per byte, with per-group scales.

    Exposes ``in_features`` / ``out_features`` / ``weight`` so it drops in as a ``LoRALinear`` base,
    matching :class:`convrot_int8.ConvRotInt8Linear` and :class:`quantize.Fp8Linear`.
    """

    is_quant_linear = True

    def __init__(self, lin: nn.Linear, rot_size: int = 256, group_size: int = GROUP_SIZE):
        super().__init__()
        rot = min(rot_size, largest_pow4_divisor(lin.in_features))
        q, scales, mins = quantize_int4_groups(rotate(lin.weight.data.float(), rot), group_size)
        self.register_buffer("qdata", pack_int4(q), persistent=False)
        # Scale and offset kept as uint8 byte views: a later ``module.to(dtype=...)`` would
        # otherwise cast them and silently change the dequantization. They are stored at
        # ``META_DTYPE`` rather than fp32 because at the default group size a 4-byte pair costs a
        # further bit per weight, while its precision is an order of magnitude finer than the 4-bit
        # grid's own error and so contributes nothing measurable.
        self.register_buffer("scales_u8", scales.view(torch.uint8), persistent=False)
        self.register_buffer("mins_u8", mins.view(torch.uint8), persistent=False)
        self.bias = (nn.Parameter(lin.bias.data.clone(), requires_grad=False)
                     if lin.bias is not None else None)
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.rot_size, self.group_size = rot, group_size

    @property
    def weight(self) -> torch.Tensor:  # device/dtype probe for LoRALinear
        return self.qdata

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """The ROTATED weight this layer applies. Still rotated: the rotation cancels against the one
        applied to the activation, so comparing it to ``lin.weight`` means rotating that first."""
        return dequantize_int4(self.qdata, self.scales_u8.view(META_DTYPE),
                               self.mins_u8.view(META_DTYPE), self.group_size, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xr = rotate(x.reshape(-1, shape[-1]), self.rot_size)
        y = convrot_int4_linear(xr, self.qdata, self.scales_u8, self.mins_u8, self.bias,
                                self.group_size)
        return y.reshape(*shape[:-1], self.out_features)


def weight_error(lin: nn.Linear, rot_size: int = 256, group_size: int = GROUP_SIZE) -> float:
    """Relative Frobenius error the 4-bit storage induces on one layer's weight.

    Measured in the ROTATED basis, which is where the quantization happens and where the error lives;
    comparing unrotated would fold in the rotation's own float round-off and report something other
    than the quantizer's error.
    """
    rot = min(rot_size, largest_pow4_divisor(lin.in_features))
    w = rotate(lin.weight.data.float(), rot)
    q, scales, mins = quantize_int4_groups(w, group_size)
    dq = dequantize_int4(pack_int4(q), scales, mins, group_size, torch.float32)
    return float((w - dq).norm() / w.norm().clamp_min(1e-12))


def eligible(lin: nn.Linear, rot_size: int = 256, group_size: int = GROUP_SIZE) -> bool:
    """Divisible by the group (also giving packing its even column count) and by the rotation block."""
    return (lin.in_features % group_size == 0 and lin.in_features % 16 == 0
            and min(rot_size, largest_pow4_divisor(lin.in_features)) >= 16)


def quantize_dit_int4(dit, rot_size: int = 256, group_size: int = GROUP_SIZE) -> int:
    """Replace the transformer blocks' ``nn.Linear`` with :class:`ConvRotInt4Linear` in place.

    Leaves ``first`` / ``last`` / text-fusion / time-MLP alone, as the fp8 and int8 paths do: they are
    small and sensitive, and at four bits that matters more, not less. Returns the number of layers
    quantized. Call BEFORE ``inject_lora``.
    """
    targets = []
    for block in dit.blocks:
        for mod in block.modules():
            for name, child in mod.named_children():
                if isinstance(child, nn.Linear) and eligible(child, rot_size, group_size):
                    targets.append((mod, name, child))
    for mod, name, child in targets:
        setattr(mod, name, ConvRotInt4Linear(child, rot_size, group_size))
    return len(targets)


__all__ = ["ConvRotInt4Linear", "quantize_dit_int4", "weight_error", "eligible", "pack_int4",
           "unpack_int4", "quantize_int4_groups", "dequantize_int4", "GROUP_SIZE"]
