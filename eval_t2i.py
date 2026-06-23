#!/usr/bin/env python
"""Base-vs-fine-tune A/B quality montage for Krea 2.

Generates the same prompts (same per-prompt seeds) with the BASE Raw model and with a
fine-tuned variant (a full-FT DiT checkpoint or a LoRA), then lays them out as a labeled
contact sheet: one row per prompt, columns [BASE | VARIANT].

  python eval_t2i.py --config config/t2i_lora.yaml \
      --lora runs/my-run/ckpts/lora_final.safetensors --out lora_eval.png
  python eval_t2i.py --config config/t2i_full.yaml \
      --ckpt runs/my-run/ckpts/dit_final.safetensors --prompts prompts.txt
  python eval_t2i.py --config config/t2i_full.yaml --out base_sanity.png   # base-only sheet

VRAM note: only ONE 12B DiT is resident at a time -- we generate all BASE images first,
then mutate the SAME dit in place (inject the LoRA, or load the ckpt state dict over it)
and regenerate. The encoder + VAE are built once and reused.
"""
from __future__ import annotations

import argparse
import os

import torch
from safetensors.torch import load_file

from loading import build_dit, build_encoder, build_vae
from lora import load_lora
from sampling import sample
from training_config import apply_runtime, dtype_of, load_config

DEFAULT_PROMPTS = [
    "a red fox walking through fresh snow, soft morning light",
    "a busy city street at night, neon signs, rain reflections",
    "a still life of fruit on a wooden table, studio lighting",
    "a portrait of an elderly fisherman, weathered face, golden hour",
    "a serene mountain lake at dawn, mist over the water",
    "a futuristic city skyline with flying vehicles, cinematic",
]


def load_prompts(path):
    if not path:
        return DEFAULT_PROMPTS
    with open(path, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {path}")
    return prompts


def gen(dit, vae, encoder, prompts, args):
    return sample(dit, vae, encoder, prompts, width=args.width, height=args.height,
                  steps=args.steps, guidance=args.guidance, seed=args.seed)


def montage(prompts, base_imgs, var_imgs, base_label, var_label, out, *, cell_h=512):
    """Contact sheet: header + one [BASE | VARIANT] row per prompt (VARIANT optional)."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    cols = [(base_label, base_imgs)] + ([(var_label, var_imgs)] if var_imgs else [])

    def thumb(im):
        return im.resize((int(im.width * cell_h / im.height), cell_h))

    thumbs = [[thumb(c[1][r]) for c in cols] for r in range(len(prompts))]
    col_w = max(t.width for row in thumbs for t in row)
    pad, hd, cs = 8, 30, 24
    W = len(cols) * col_w + (len(cols) + 1) * pad
    H = hd + len(prompts) * (cs + cell_h + pad) + pad

    sheet = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    # column headers
    d.rectangle([0, 0, W, hd], fill=(20, 20, 20))
    for c, (label, _) in enumerate(cols):
        d.text((pad + c * (col_w + pad), 7), label, fill=(255, 255, 255), font=font)
    # rows
    y = hd + pad
    for r, prompt in enumerate(prompts):
        d.rectangle([0, y, W, y + cs], fill=(40, 40, 40))
        d.text((pad, y + 4), prompt[:140], fill=(220, 220, 220), font=font)
        y += cs
        for c in range(len(cols)):
            sheet.paste(thumbs[r][c], (pad + c * (col_w + pad), y))
        y += cell_h + pad
    sheet.save(out)
    print("saved", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=None, help="txt file, one prompt per line (default: built-in set)")
    ap.add_argument("--ckpt", default=None, help="fine-tuned DiT state_dict (.safetensors)")
    ap.add_argument("--lora", default=None, help="trained LoRA adapters (.safetensors)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_montage.png")
    args = ap.parse_args()
    if args.ckpt and args.lora:
        raise SystemExit("pass at most one of --ckpt / --lora")

    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)
    prompts = load_prompts(args.prompts)

    # BASE column: always the Raw model.
    dit = build_dit(cfg, device, dtype, load_weights=True, train=False)
    encoder = build_encoder(cfg, device, dtype, train=False)
    vae = build_vae(cfg, device, dtype)
    print(f"generating BASE: {len(prompts)} prompts", flush=True)
    base_imgs = gen(dit, vae, encoder, prompts, args)

    # VARIANT column: mutate the same dit in place (no second 12B model resident).
    var_imgs, var_label = None, None
    if args.lora:
        load_lora(dit, args.lora)
        dit = dit.to(device=device, dtype=dtype).eval()
        var_label = f"LoRA: {os.path.basename(args.lora)}"
    elif args.ckpt:
        dit.load_state_dict(load_file(args.ckpt), strict=True)
        dit = dit.to(device=device, dtype=dtype).eval()
        var_label = f"ckpt: {os.path.basename(args.ckpt)}"
    if var_label:
        torch.cuda.empty_cache()
        print(f"generating VARIANT ({var_label})", flush=True)
        var_imgs = gen(dit, vae, encoder, prompts, args)

    montage(prompts, base_imgs, var_imgs, "BASE (raw)", var_label, args.out)


if __name__ == "__main__":
    main()
