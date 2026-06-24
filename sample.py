#!/usr/bin/env python
"""Standalone text-to-image generation for Krea 2: base, a fine-tuned DiT checkpoint, or + a LoRA.

  python sample.py --config config/t2i_full.yaml --prompt "a fox in the snow" --out fox.png
  python sample.py --config ... --ckpt runs/r/ckpts/dit_final.safetensors --prompt "..."
  python sample.py --config ... --lora runs/r/ckpts/lora_final.safetensors --prompt "..."

Recipe (matches the reference inference.py, which uses the same sampler):
  * Raw   : ~52 steps, --guidance 3.5   (undistilled base, full CFG)
  * Turbo : --base krea/Krea-2-Turbo --base-file <turbo>.safetensors --steps 8 --guidance 0 --mu 1.15
    -> the recommended "train a LoRA on Raw, run it on Turbo" path: load the Turbo base + your LoRA.
"""
from __future__ import annotations

import argparse
import os

import torch
from PIL import Image
from safetensors.torch import load_file

from loading import build_dit, build_encoder, build_vae
from lora import load_lora, lora_scaled
from sampling import sample
from training_config import apply_runtime, dtype_of, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--ckpt", default=None, help="fine-tuned DiT state_dict (.safetensors)")
    ap.add_argument("--lora", default=None, help="trained LoRA adapters (.safetensors)")
    ap.add_argument("--lora-scale", dest="lora_scale", type=float, default=1.0,
                    help="runtime adapter strength (slider knob): 1.0 normal, 0 off, <0 inverts")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5, help="CFG scale (0 disables CFG; use 0 for Turbo)")
    ap.add_argument("--mu", type=float, default=None,
                    help="pin the resolution-shift mu (Turbo: 1.15); default None -> interpolate from y1/y2")
    ap.add_argument("--y1", type=float, default=0.5, help="mu at min resolution")
    ap.add_argument("--y2", type=float, default=1.15, help="mu at max resolution")
    ap.add_argument("--base", default=None,
                    help="override the base DiT repo/dir (e.g. krea/Krea-2-Turbo) -> run a Raw-trained LoRA on Turbo")
    ap.add_argument("--base-file", dest="base_file", default=None, help="weight filename within --base")
    ap.add_argument("--style-ref", dest="style_ref", default=None,
                    help="a style/reference image fed through the Qwen3-VL encoder (needs a DiT/LoRA trained for image conditioning)")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num", type=int, default=1)
    ap.add_argument("--out", default="sample.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)
    if args.base:           # load a different base (e.g. Turbo) before applying a Raw-trained LoRA
        cfg.paths.dit_repo = args.base
    if args.base_file:
        cfg.paths.dit_file = args.base_file

    dit = build_dit(cfg, device, dtype, load_weights=(args.ckpt is None), train=False)
    if args.ckpt:
        dit.load_state_dict(load_file(args.ckpt), strict=True, assign=False)
        dit = dit.to(device=device, dtype=dtype).eval()
    if args.lora:
        load_lora(dit, args.lora)
        dit = dit.to(device=device, dtype=dtype).eval()
    encoder = build_encoder(cfg, device, dtype, train=False)
    vae = build_vae(cfg, device, dtype)

    prompts = [args.prompt] * args.num
    negs = [args.negative] * args.num if args.negative else None
    style_imgs = [Image.open(args.style_ref).convert("RGB")] * args.num if args.style_ref else None
    with lora_scaled(dit, args.lora_scale):  # no-op when no LoRA / scale 1.0; the slider knob otherwise
        imgs = sample(dit, vae, encoder, prompts, negative_prompts=negs, width=args.width,
                      height=args.height, steps=args.steps, guidance=args.guidance, seed=args.seed,
                      mu=args.mu, y1=args.y1, y2=args.y2, images=style_imgs)
    if args.num == 1:
        imgs[0].save(args.out)
        print("saved", args.out, flush=True)
    else:
        base, ext = os.path.splitext(args.out)
        for i, im in enumerate(imgs):
            im.save(f"{base}_{i}{ext}")
        print(f"saved {args.num} images -> {base}_*{ext}", flush=True)


if __name__ == "__main__":
    main()
