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


def _act_quant_padded_torch(x: torch.Tensor, qmax: int):
    """Per-token activation quantization, rows padded to a multiple of 32 for ``torch._int_mm``."""
    q, scales = quantize_int8_rows(x, qmax)
    rows = q.shape[0]
    rows_pad = -(-rows // 32) * 32
    if rows_pad != rows:
        q = F.pad(q, (0, 0, 0, rows_pad - rows))
        scales = F.pad(scales, (0, rows_pad - rows), value=1.0)
    return q, scales


def _epilogue_torch(i32, a_scales, w_scales, bias, out_dtype: torch.dtype) -> torch.Tensor:
    """``out = i32 * a_scales[:, None] * w_scales[None, :] (+ bias)``, accumulated in fp32."""
    out = i32.float() * w_scales
    out = out * a_scales.unsqueeze(1)
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


# --------------------------------------------------------------------------- #
# Fused kernels for the two stages around the matmul. Expressed as eager elementwise ops, each
# stage walks a full-size tensor several times in fp32; one kernel per side collapses that into a
# single pass. Both fall back to the torch implementations above when Triton is unavailable.
# --------------------------------------------------------------------------- #
_triton_ok: bool | None = None


def _have_triton() -> bool:
    global _triton_ok
    if _triton_ok is None:
        try:
            import triton  # noqa: F401
            import triton.language  # noqa: F401

            _triton_ok = True
        except Exception:
            _triton_ok = False
    return _triton_ok


def _build_kernels():
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice

        _round = libdevice.rint          # half-to-even, matching torch.round
    except Exception:                     # older Triton: floor(x + 0.5) away from zero
        def _round(v):
            return tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))

    @triton.jit
    def act_quant(x_ptr, q_ptr, s_ptr, M, K, QMAX: tl.constexpr, BLOCK_K: tl.constexpr):
        """Row amax -> scale -> int8 codes in one launch. Rows past ``M`` are the padding
        ``torch._int_mm`` needs and are written as zeros with a unit scale."""
        row = tl.program_id(0)
        if row >= M:
            for k0 in range(0, K, BLOCK_K):
                offs = k0 + tl.arange(0, BLOCK_K)
                tl.store(q_ptr + row * K + offs, tl.zeros((BLOCK_K,), tl.int8), mask=offs < K)
            tl.store(s_ptr + row, 1.0)
        else:
            amax = tl.zeros((BLOCK_K,), tl.float32)
            for k0 in range(0, K, BLOCK_K):
                offs = k0 + tl.arange(0, BLOCK_K)
                x = tl.load(x_ptr + row * K + offs, mask=offs < K, other=0.0).to(tl.float32)
                amax = tl.maximum(amax, tl.abs(x))
            s = tl.max(amax) / QMAX
            s = tl.where(s > 0, s, 1.0)
            tl.store(s_ptr + row, s)
            for k0 in range(0, K, BLOCK_K):
                offs = k0 + tl.arange(0, BLOCK_K)
                x = tl.load(x_ptr + row * K + offs, mask=offs < K, other=0.0).to(tl.float32)
                q = _round(x / s)
                q = tl.minimum(tl.maximum(q, -QMAX), QMAX)
                tl.store(q_ptr + row * K + offs, q.to(tl.int8), mask=offs < K)

    @triton.jit
    def epilogue(i32_ptr, a_ptr, w_ptr, b_ptr, out_ptr, N,
                 HAS_BIAS: tl.constexpr, BLOCK_N: tl.constexpr):
        """``out = i32 * a_scale[row] * w_scale[col] (+ bias)``, one pass, written in out dtype."""
        row, blk = tl.program_id(0), tl.program_id(1)
        offs = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        acc = tl.load(i32_ptr + row * N + offs, mask=mask, other=0).to(tl.float32)
        acc = acc * tl.load(a_ptr + row) * tl.load(w_ptr + offs, mask=mask, other=0.0)
        if HAS_BIAS:
            acc = acc + tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * N + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)

    return act_quant, epilogue


_kernels = None


def _get_kernels():
    global _kernels
    if _kernels is None:
        _kernels = _build_kernels()
    return _kernels


def _act_quant_padded(x: torch.Tensor, qmax: int):
    if not (_have_triton() and x.is_cuda):
        return _act_quant_padded_torch(x, qmax)
    m, k = x.shape
    m_pad = -(-m // 32) * 32
    q = torch.empty(m_pad, k, device=x.device, dtype=torch.int8)
    s = torch.empty(m_pad, device=x.device, dtype=torch.float32)
    act_quant, _ = _get_kernels()
    act_quant[(m_pad,)](x, q, s, m, k, QMAX=qmax, BLOCK_K=1024, num_warps=8)
    return q, s


def _epilogue(i32, a_scales, w_scales, bias, out_dtype: torch.dtype) -> torch.Tensor:
    if not (_have_triton() and i32.is_cuda):
        return _epilogue_torch(i32, a_scales, w_scales, bias, out_dtype)
    import triton

    m, n = i32.shape
    out = torch.empty(m, n, device=i32.device, dtype=out_dtype)
    _, epilogue = _get_kernels()
    epilogue[(m, triton.cdiv(n, 1024))](i32, a_scales, w_scales,
                                        bias if bias is not None else a_scales, out, n,
                                        HAS_BIAS=bias is not None, BLOCK_N=1024, num_warps=4)
    return out


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
