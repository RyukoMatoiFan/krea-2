#!/usr/bin/env python
"""Average the tail of a run's checkpoints into one model.

Not a speedup: it does not change training and costs nothing at run time. It is a way to reach a
given quality at a lower step count, by averaging away the noise that SGD leaves swinging around the
minimum. The gain is real but not free of conditions — it needs the checkpoints to be near a common
basin, which in practice means the LR was already low (or the schedule annealed) and the spacing is
not so wide that the weights have genuinely moved apart. So it is validated against the last-step
model rather than assumed, which is what ``eval_t2i.py`` is for.

Averaging is done in fp32 and cast back at the end: summing K bf16 tensors in bf16 loses low bits on
every add, and the whole point is to keep the small differences between checkpoints.

    python average_checkpoints.py --ckpt-dir runs/<run>/ckpts --pattern "dit_step*.safetensors" \\
        --last 3 --out runs/<run>/ckpts/dit_avg3.safetensors
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import torch
from safetensors.torch import load_file, save_file


def step_of(path: str) -> int:
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--pattern", default="dit_step*.safetensors")
    ap.add_argument("--last", type=int, default=3, help="how many of the newest to average")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth checkpoint from the tail; widens the window without "
                         "needing more of them retained")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.ckpt_dir, args.pattern)), key=step_of)
    if not files:
        raise SystemExit(f"no checkpoints matching {args.pattern} in {args.ckpt_dir}")
    picked = files[::-1][:: max(1, args.stride)][: args.last][::-1]
    if len(picked) < 2:
        raise SystemExit(f"need at least 2 checkpoints to average, found {len(picked)}")
    print("averaging %d checkpoints (steps %s)" % (
        len(picked), ", ".join(str(step_of(p)) for p in picked)), flush=True)

    acc, dtypes = {}, {}
    for i, p in enumerate(picked):
        sd = load_file(p)
        if i == 0:
            for k, v in sd.items():
                dtypes[k] = v.dtype
                acc[k] = v.to(torch.float32)
        else:
            if set(sd) != set(acc):
                missing = (set(acc) ^ set(sd))
                raise SystemExit(f"{os.path.basename(p)} has a different key set "
                                 f"({len(missing)} differing, e.g. {sorted(missing)[:2]}) — "
                                 "refusing to average checkpoints from different runs")
            for k, v in sd.items():
                acc[k] += v.to(torch.float32)
        del sd
        print("  + %s" % os.path.basename(p), flush=True)

    n = float(len(picked))
    out = {k: (v / n).to(dtypes[k]) for k, v in acc.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_file(out, args.out)
    print("wrote %s (%.2f GB)" % (args.out, os.path.getsize(args.out) / 1e9))
    print("Validate before shipping: render the averaged model against the last-step one at the "
          "same seeds and prompts (eval_t2i.py --ckpt ...). Averaging can hurt.")


if __name__ == "__main__":
    main()
