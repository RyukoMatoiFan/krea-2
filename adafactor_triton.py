#!/usr/bin/env python
"""Triton-fused Adafactor update for the fused-backward hook.

The update is memory-traffic bound: a straightforward implementation materialises a chain of
intermediates -- an fp32 copy of the gradient, the squared update, the update itself, an fp32 copy of
the parameter and a randomness buffer for stochastic rounding -- each the size of the parameter.

This version never materialises anything of parameter size. ``upd`` is recomputed from ``g`` instead
of being stored, trading a cheap re-read for three large writes:

    pass 1   read g                  -> row/col sums of g^2
    pass 2   read g, p               -> sum(upd^2), sum(p^2)
    pass 3   read g, p, write p      -> delta + stochastic round

Adafactor needs three sequential reductions that cannot be collapsed (the row/column second moments,
then the update RMS for clipping, then the parameter RMS for the relative step size), so three
passes is the minimum; the saving comes from not storing the intermediates between them.

The fast path covers 2D parameters, which hold essentially all of the DiT's elements (linear
weights). 1D parameters take Adafactor's unfactored branch and anything with more dimensions is rare
and small, so both fall back to the eager reference -- correctness first, and the traffic they
represent is negligible.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.optim import AdamW

from fused_adamw import step_adafactor_parameter

EPS1, EPS2 = 1e-30, 1e-3
CLIP_THRESHOLD, DECAY_RATE = 1.0, 0.8


@triton.jit
def _rowcol_sumsq_kernel(g_ptr, row_ptr, col_ptr, M, N, stride_m, stride_n,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Accumulate per-row and per-column sums of g^2 in one pass over the gradient."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m, mask_n = offs_m < M, offs_n < N
    ptrs = g_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    g = tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    g2 = g * g
    tl.atomic_add(row_ptr + offs_m, tl.sum(g2, axis=1), mask=mask_m)
    tl.atomic_add(col_ptr + offs_n, tl.sum(g2, axis=0), mask=mask_n)


@triton.jit
def _reduce_kernel(g_ptr, p_ptr, r_ptr, c_ptr, upd2_ptr, p2_ptr, M, N, stride_m, stride_n,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Second pass: sum(upd^2) for the RMS clip and sum(p^2) for the parameter-scaled LR.

    ``upd`` is computed on the fly from ``g`` and the row/column factors, never stored.
    """
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m, mask_n = offs_m < M, offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    off = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    g = tl.load(g_ptr + off, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs_m, mask=mask_m, other=0.0)
    c = tl.load(c_ptr + offs_n, mask=mask_n, other=0.0)
    upd = g * r[:, None] * c[None, :]
    tl.atomic_add(upd2_ptr, tl.sum(tl.where(mask, upd * upd, 0.0)))
    tl.atomic_add(p2_ptr, tl.sum(tl.where(mask, p * p, 0.0)))


@triton.jit
def _apply_kernel(g_ptr, p_ptr, r_ptr, c_ptr, clip_ptr, lrt_ptr, wd_scale, seed,
                  M, N, stride_m, stride_n, STOCHASTIC: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Third pass: recompute upd, form the delta, apply it to the bf16 weight in place.

    ``clip`` and ``lrt`` arrive as device scalars so no host synchronisation is needed.
    """
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m, mask_n = offs_m < M, offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    off = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    g = tl.load(g_ptr + off, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(p_ptr + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs_m, mask=mask_m, other=0.0)
    c = tl.load(c_ptr + offs_n, mask=mask_n, other=0.0)
    clip = tl.load(clip_ptr)
    lrt = tl.load(lrt_ptr)

    upd = g * r[:, None] * c[None, :] / clip
    delta = upd * (-lrt) - p * wd_scale
    out = p + delta

    if STOCHASTIC:
        # Round-to-nearest would bias a bf16 weight that is updated millions of times; adding 16
        # random low bits before truncating makes the rounding unbiased in expectation.
        bits = out.to(tl.int32, bitcast=True)
        # The cast back to int32 is required: masking with 0xFFFF makes the result unsigned, and
        # Triton refuses to combine an unsigned tensor with the negative truncation mask.
        noise = (tl.randint(seed, off) & 0xFFFF).to(tl.int32)
        out = ((bits + noise) & -65536).to(tl.float32, bitcast=True)
    tl.store(p_ptr + off, out.to(p_ptr.dtype.element_ty), mask=mask)


def _grid(M, N, bm, bn):
    return (triton.cdiv(M, bm), triton.cdiv(N, bn))


@torch.no_grad()
def triton_adafactor_step(p: Tensor, g: Tensor, row: Tensor, col: Tensor, *,
                          beta2t: float, lr: float, wd: float, stochastic: bool,
                          seed: int, block_m: int = 32, block_n: int = 128) -> None:
    """One Adafactor step for a 2D parameter, in place. ``g`` may be bf16 (never upcast in bulk)."""
    M, N = p.shape
    sm, sn = p.stride()
    grid = _grid(M, N, block_m, block_n)

    row_sum = torch.zeros_like(row)
    col_sum = torch.zeros_like(col)
    _rowcol_sumsq_kernel[grid](g, row_sum, col_sum, M, N, sm, sn,
                               BLOCK_M=block_m, BLOCK_N=block_n)
    # State update and the row/column factors are tiny (M+N elements) -- eager is fine here.
    row.mul_(beta2t).add_(row_sum / N + EPS1, alpha=1.0 - beta2t)
    col.mul_(beta2t).add_(col_sum / M + EPS1, alpha=1.0 - beta2t)
    r_factor = (row / row.mean()).rsqrt()
    c_factor = col.rsqrt()

    upd2 = torch.zeros((), device=p.device, dtype=torch.float32)
    p2 = torch.zeros((), device=p.device, dtype=torch.float32)
    _reduce_kernel[grid](g, p, r_factor, c_factor, upd2, p2, M, N, sm, sn,
                         BLOCK_M=block_m, BLOCK_N=block_n)

    n_el = M * N
    rms = (upd2 / n_el).sqrt()
    clip = torch.clamp(rms / CLIP_THRESHOLD, min=1.0)
    p_rms = (p2 / n_el).sqrt()
    lrt = lr * torch.clamp(p_rms, min=EPS2)

    _apply_kernel[grid](g, p, r_factor, c_factor, clip, lrt, lr * wd, seed, M, N, sm, sn,
                        STOCHASTIC=bool(stochastic and p.dtype == torch.bfloat16),
                        BLOCK_M=block_m, BLOCK_N=block_n)


@torch.no_grad()
def step_adafactor_parameter_triton(self, p: Tensor, group: dict, i: int) -> None:
    """Drop-in for ``fused_adamw.step_adafactor_parameter``; 2D params take the Triton path."""
    if p.grad is None:
        return
    if p.grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients")
    if (p.dim() != 2 or group.get("maximize") or not p.is_cuda
            or not p.is_contiguous() or not p.grad.is_contiguous()):
        step_adafactor_parameter(self, p, group, i)
        return

    state = self.state[p]
    if len(state) == 0:
        state["step"] = 0
        state["exp_avg_sq_row"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)
        state["exp_avg_sq_col"] = torch.zeros(p.shape[1], device=p.device, dtype=torch.float32)
    state["step"] += 1
    self._sr_seed = getattr(self, "_sr_seed", 0) + 1

    triton_adafactor_step(
        p, p.grad, state["exp_avg_sq_row"], state["exp_avg_sq_col"],
        beta2t=1.0 - state["step"] ** (-DECAY_RATE), lr=group["lr"],
        wd=float(group.get("weight_decay") or 0.0),
        stochastic=getattr(self, "stochastic_rounding", False), seed=self._sr_seed)


def build_triton_adafactor(params, lr: float, *, weight_decay: float = 0.0,
                           stochastic_rounding: bool = True) -> AdamW:
    """Same contract as ``fused_adamw.build_fused_adafactor``, Triton update for 2D parameters."""
    opt = AdamW(params, lr=lr, weight_decay=weight_decay, foreach=False, fused=False)
    opt.stochastic_rounding = stochastic_rounding
    opt.offload_states = False
    opt._sr_seed = 0
    opt.step_parameter = step_adafactor_parameter_triton.__get__(opt, AdamW)
    return opt
