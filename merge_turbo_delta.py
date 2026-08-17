#!/usr/bin/env python
"""Give a full fine-tune few-step sampling by merging in the distillation delta.

A fine-tune trained on Raw samples like Raw: ~52 steps. Turbo samples in 8, but is a *different*
checkpoint, so a fine-tune cannot simply be "applied" to it. What transfers is the difference:

    delta = W_turbo - W_raw          <- the distillation, someone else already computed it
    W_out = W_ft + scale * rank_r(delta)

Note which side gets compressed. The low-rank approximation lands on the delta -- a fixed artifact we
did not train -- while the fine-tune keeps full rank and full fidelity. Doing it the other way
(compressing the fine-tune and adding it to Turbo) spends precision on exactly the weights the run
was for.

One SVD serves every rank: factors are truncations of the same decomposition, so ``--cache`` makes a
rank sweep nearly free after the first pass.

Self-test, which needs no fine-tune and so can run before one exists: with ``--ft`` equal to
``--raw``, ``--scale 1`` and ``--rank 0``, the output must reproduce Turbo exactly.

    python merge_turbo_delta.py --raw raw.safetensors --turbo turbo.safetensors \\
        --ft dit_step020000.safetensors --rank 512 --scale 1.0 --out merged.safetensors
    python merge_turbo_delta.py --raw raw.safetensors --turbo turbo.safetensors --verify
"""
from __future__ import annotations

import argparse
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Low-rank factorisation only makes sense for matrices. Norms, biases and 1-D scales are tiny and
# compress badly -- their delta is applied verbatim regardless of --rank.
MIN_MATRIX_DIM = 32


def is_matrix(t: torch.Tensor) -> bool:
    return t.ndim == 2 and min(t.shape) >= MIN_MATRIX_DIM


def low_rank(delta: torch.Tensor, rank: int, device: str) -> torch.Tensor:
    """Rank-r approximation of ``delta`` via randomised SVD.

    float32 for the decomposition: bf16 has ~8 bits of mantissa, and the singular values of a weight
    delta span several orders of magnitude, so factorising in the storage dtype loses the tail that
    the approximation is supposed to keep.
    """
    a = delta.to(device=device, dtype=torch.float32)
    q = min(rank, min(a.shape))
    u, s, v = torch.svd_lowrank(a, q=q, niter=4)
    return (u * s) @ v.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="base checkpoint the fine-tune started from")
    ap.add_argument("--turbo", required=True, help="distilled checkpoint")
    ap.add_argument("--ft", default=None, help="fine-tuned DiT; defaults to --raw (self-test)")
    ap.add_argument("--out", default=None, help="written unless --verify")
    ap.add_argument("--scale", type=float, default=1.0, help="delta multiplier")
    ap.add_argument("--rank", type=int, default=0, help="0 = full-rank delta, no factorisation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify", action="store_true",
                    help="compare the merge against --turbo instead of writing it; with the default "
                         "--ft, --scale 1 and --rank 0 the two must be identical")
    args = ap.parse_args()

    ft_path = args.ft or args.raw
    self_test = (ft_path == args.raw and args.scale == 1.0 and args.rank == 0)
    if args.verify and not self_test:
        print("[warn] --verify is only exact for --ft==--raw, --scale 1, --rank 0; "
              "reporting differences anyway", flush=True)
    if not args.verify and not args.out:
        raise SystemExit("--out is required unless --verify")

    fr = safe_open(args.raw, framework="pt", device="cpu")
    ftb = safe_open(args.turbo, framework="pt", device="cpu")
    ff = safe_open(ft_path, framework="pt", device="cpu")

    raw_keys, turbo_keys, ft_keys = set(fr.keys()), set(ftb.keys()), set(ff.keys())
    if raw_keys != turbo_keys:
        # Raw and Turbo must be the same architecture for a delta to mean anything at all.
        only_r, only_t = raw_keys - turbo_keys, turbo_keys - raw_keys
        raise SystemExit(f"raw/turbo key mismatch: {len(only_r)} only in raw, {len(only_t)} only in "
                         f"turbo (e.g. {sorted(only_r)[:2]} / {sorted(only_t)[:2]})")
    missing = raw_keys - ft_keys
    if missing:
        raise SystemExit(f"fine-tune is missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")

    out = {}
    n_lr = n_direct = 0
    worst = 0.0
    worst_key = ""
    total_abs = 0.0
    n_elem = 0

    for k in sorted(raw_keys):
        w_raw, w_turbo, w_ft = fr.get_tensor(k), ftb.get_tensor(k), ff.get_tensor(k)
        delta = w_turbo.to(torch.float32) - w_raw.to(torch.float32)
        if args.rank > 0 and is_matrix(delta):
            d = low_rank(delta, args.rank, args.device).cpu()
            n_lr += 1
        else:
            d = delta
            n_direct += 1
        merged = (w_ft.to(torch.float32) + args.scale * d).to(w_ft.dtype)

        if args.verify:
            diff = (merged.to(torch.float32) - w_turbo.to(torch.float32)).abs()
            m = diff.max().item()
            total_abs += diff.sum().item()
            n_elem += diff.numel()
            if m > worst:
                worst, worst_key = m, k
        else:
            out[k] = merged

    if args.verify:
        print(f"tensors {len(raw_keys):,}  low-rank {n_lr:,}  direct {n_direct:,}")
        print(f"max |merged - turbo| = {worst:.3e}   at {worst_key}")
        print(f"mean |merged - turbo| = {total_abs / max(1, n_elem):.3e}")
        if self_test:
            # Not "exactly zero": raw + (turbo - raw) is exact in real arithmetic but not in
            # floating point, and fp32-stored tensors leave cancellation residue around 1e-10.
            # bf16 tensors do round-trip exactly, so the tolerance only ever matters for fp32.
            ok = worst < 1e-6
            print("SELF-TEST " + ("PASS: merge reproduces Turbo to floating-point precision"
                                  if ok else f"FAIL: max diff {worst:.3e} exceeds 1e-6 at {worst_key}"))
            raise SystemExit(0 if ok else 1)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_file(out, args.out)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e9:.2f} GB)  "
          f"scale={args.scale} rank={args.rank or 'full'}  "
          f"low-rank {n_lr:,} / direct {n_direct:,}")


if __name__ == "__main__":
    main()
