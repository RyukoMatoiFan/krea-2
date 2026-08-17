#!/usr/bin/env python
"""Prove a GPU backend is usable for this trainer before spending days on it.

Written for bringing the trainer up on ROCm, but every check is vendor-neutral and passes on CUDA,
which is the point: the CUDA run is the control. The ladder is ordered cheapest-first and stops at
the first failure, so a broken environment costs seconds rather than surfacing as a disappointing
loss curve on day three.

The rungs escalate from "does it import" to the only question that really matters -- **does the
optimizer produce the same numbers as the reference implementation**. A backend that crashes is a
good outcome; a backend that quietly computes something else is the one that wastes a week.

    python rocm_verify.py            # full ladder
    python rocm_verify.py --quick    # skip the slower numerical rungs
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []


def rung(name):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                status, detail = fn(*a, **k)
            except Exception as e:
                status, detail = FAIL, f"{type(e).__name__}: {e}"
                if os.environ.get("ROCM_VERIFY_TRACE"):
                    traceback.print_exc()
            results.append((name, status, detail))
            print(f"[{status}] {name}: {detail}", flush=True)
            return status
        return wrapped
    return deco


@rung("R0 environment")
def r0():
    hip, cuda = torch.version.hip, torch.version.cuda
    if not torch.cuda.is_available():
        return FAIL, "torch.cuda.is_available() is False -- wrong wheel for this host"
    dev = torch.cuda.get_device_name(0)
    return PASS, (f"torch {torch.__version__} hip={hip} cuda={cuda} device={dev!r} "
                  f"cap={torch.cuda.get_device_capability()}")


@rung("R0b cuDNN-pin guard resolves correctly")
def r0b():
    import mmdit
    from torch.nn.attention import SDPBackend
    enum_exists = hasattr(SDPBackend, "CUDNN_ATTENTION")
    on_hip = torch.version.hip is not None
    # The trap this guards: the enum member exists on EVERY build and cuda.is_available() is True on
    # ROCm, so a naive check pins a backend that does not exist there.
    want = enum_exists and not on_hip
    if mmdit._CUDNN_SDPA != want:
        return FAIL, f"_CUDNN_SDPA={mmdit._CUDNN_SDPA}, expected {want} (enum={enum_exists}, hip={on_hip})"
    return PASS, f"_CUDNN_SDPA={mmdit._CUDNN_SDPA} (enum present={enum_exists}, hip={on_hip})"


@rung("R1 triton compiles for this target")
def r1():
    import triton
    target = triton.runtime.driver.active.get_current_target()
    backends = list(getattr(triton.backends, "backends", {}).keys())
    return PASS, f"triton {triton.__version__} backends={backends} target={target}"


@rung("R2 fp32 atomics accumulate")
def r2():
    import adafactor_triton as A
    A._ATOMICS_VERIFIED = False
    A._verify_atomics(torch.device("cuda"))
    # Report BOTH names. The ROCm build reads PYTORCH_HIP_ALLOC_CONF; printing only the CUDA name
    # would show "<unset>" on a correctly configured AMD host and send someone chasing this rung.
    alloc = ", ".join(f"{k}={os.environ.get(k, '<unset>')}"
                      for k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"))
    return PASS, f"exact counts under {alloc}"


def _attn_probe(masked: bool):
    """Run one attention fwd+bwd at production shapes and report peak memory and time.

    Memory is the criterion, not time. A fused kernel never materialises the score matrix; the math
    fallback keeps one of B*H*L*L*2 bytes per call. The two states are separated by orders of
    magnitude, far outside any timing ambiguity.

    Backward is included because it has separate backend eligibility -- a forward-only probe can
    pass while the training step falls back.
    """
    B, H, KV, L, D = 1, 48, 12, 4608, 128
    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, KV, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    mask = None
    if masked:
        # The production hot path. Text is bucketed to a multiple of 128, so the mask is NOT
        # all-true on most steps and a dense (B,1,1,L) mask reaches the kernel.
        mask = torch.ones(B, 1, 1, L, dtype=torch.bool, device="cuda")
        mask[..., -88:] = False
    f = torch.nn.functional.scaled_dot_product_attention

    f(q, k, v, attn_mask=mask, enable_gqa=True).sum().backward()   # warm
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    import time
    t = time.perf_counter()
    f(q, k, v, attn_mask=mask, enable_gqa=True).sum().backward()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1e3
    peak_mb = (torch.cuda.max_memory_allocated() - base) / 1e6
    score_mb = B * H * L * L * 2 / 1e6
    return ms, peak_mb, score_mb


@rung("R3 attention is fused, not the math fallback (unmasked)")
def r3():
    ms, peak_mb, score_mb = _attn_probe(masked=False)
    # Threshold sits far below one score matrix and far above legitimate fused workspace.
    if peak_mb > 0.5 * score_mb:
        return FAIL, (f"peak {peak_mb:.0f} MB for fwd+bwd -- a score matrix is {score_mb:.0f} MB, "
                      f"so this is the math fallback ({ms:.0f} ms)")
    return PASS, f"peak {peak_mb:.0f} MB (score matrix would be {score_mb:.0f} MB), {ms:.0f} ms"


@rung("R3b masked attention is fused (the production path)")
def r3b():
    """Separate rung on purpose: masking and GQA have independent backend eligibility, and with
    bucketed text the masked path is what almost every training step actually executes. A green
    unmasked rung says nothing about it."""
    ms, peak_mb, score_mb = _attn_probe(masked=True)
    if peak_mb > 0.5 * score_mb:
        return FAIL, (f"MASKED path fell back: peak {peak_mb:.0f} MB vs a {score_mb:.0f} MB score "
                      f"matrix ({ms:.0f} ms). This is the hot path -- the run would be silently slow.")
    return PASS, f"peak {peak_mb:.0f} MB (score matrix would be {score_mb:.0f} MB), {ms:.0f} ms"


@rung("R4 eager attention path runs")
def r4():
    import mmdit
    q = torch.randn(1, 48, 1024, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 12, 1024, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = mmdit.attention(q, k, v, gqa=True)          # NOT under torch.compile: exercises the pin
    if not torch.isfinite(out).all():
        return FAIL, "non-finite output"
    return PASS, f"out {tuple(out.shape)} finite"


@rung("R5 triton optimizer == eager reference")
def r5():
    """The rung that catches a SILENTLY WRONG update.

    Runs one Adafactor step through the Triton kernel and through the eager reference on identical
    inputs and compares the resulting parameters. Stochastic rounding is disabled so the two are
    deterministic and directly comparable -- with it on, the two differ by design and the test
    would prove nothing.
    """
    import adafactor_triton as A
    from fused_adamw import step_adafactor_parameter

    torch.manual_seed(0)
    M, N = 1024, 2048
    p_ref = torch.randn(M, N, device="cuda", dtype=torch.float32) * 0.02
    g = torch.randn(M, N, device="cuda", dtype=torch.float32) * 1e-3
    p_tri = p_ref.clone()

    group = {"lr": 1e-5, "weight_decay": 0.0, "eps": (1e-30, 1e-3), "clip_threshold": 1.0,
             "decay_rate": -0.8, "beta1": None, "maximize": False, "stochastic": False}

    # defaultdict, not a plain dict: both implementations do `state = self.state[p]` and rely on
    # the optimizer's usual auto-vivifying state map to create the entry on first touch.
    import collections

    class _Opt:
        def __init__(self):
            self.state = collections.defaultdict(dict)
    ref_opt, tri_opt = _Opt(), _Opt()

    p0 = p_ref.clone()                       # keep the pre-update weights: the yardstick
    p_ref.grad = g.clone()
    p_tri.grad = g.clone()
    step_adafactor_parameter(ref_opt, p_ref, group, 0)
    A.step_adafactor_parameter_triton(tri_opt, p_tri, group, 0)

    d = (p_ref - p_tri).abs().max().item()
    upd = (p_ref - p0).abs().max().item()     # how far the reference actually moved
    if upd <= 0.0:
        return FAIL, "reference update did nothing -- the comparison would be vacuous"
    rel = d / upd
    # Tolerance is relative to the UPDATE, not the weight: the interesting failure is an update with
    # the wrong scale, which a weight-relative tolerance would hide (updates are ~1e-5 of a weight).
    if not torch.isfinite(p_tri).all():
        return FAIL, "triton path produced non-finite parameters"
    if rel > 0.02:
        return FAIL, (f"max|ref-triton| = {d:.3e} against an update of {upd:.3e} "
                      f"= {rel:.1%} of it -- too large")
    return PASS, f"max diff {d:.2e} vs update {upd:.2e} = {rel:.3%} of the update"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="environment and kernel rungs only")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ladder = [r0, r0b, r1, r2, r3, r3b, r4] + ([] if args.quick else [r5])
    for fn in ladder:
        if fn() == FAIL:
            print("\nstopping at first failure -- later rungs assume this one passed", flush=True)
            break

    print("\n" + "=" * 62)
    for name, status, _ in results:
        print(f"  {status:4}  {name}")
    bad = [n for n, s, _ in results if s == FAIL]
    print("=" * 62)
    print("VERDICT:", "USABLE" if not bad else f"NOT USABLE ({len(bad)} failed)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
