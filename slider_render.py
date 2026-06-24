#!/usr/bin/env python
"""Render a trained Krea 2 slider LoRA across a list of strengths (loads the model ONCE).

  python slider_render.py --lora <slider>.safetensors --scales " -3,0,2,3.5" --out sweep.png
"""
import argparse
import os

import torch
from PIL import Image, ImageDraw

from loading import build_dit, build_encoder, build_vae
from lora import load_lora, lora_scaled
from sampling import sample
from training_config import apply_runtime, dtype_of, load_config

PROMPTS = [
    "a close-up portrait of a person's face",
    "a close-up of a flower with water droplets on the petals",
    "a landscape with trees, a house and a path",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/slider.yaml")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--scales", default="-3,0,2,3.5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    scales = [float(x) for x in args.scales.split(",") if x.strip() != ""]
    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)
    dit = build_dit(cfg, device, dtype, load_weights=True, train=False)
    load_lora(dit, args.lora)
    dit = dit.to(device=device, dtype=dtype).eval()
    encoder = build_encoder(cfg, device, dtype, train=False)
    vae = build_vae(cfg, device, dtype)

    rows = []
    for pi, prompt in enumerate(PROMPTS):
        panels = []
        for sc in scales:
            with lora_scaled(dit, sc):
                panels.append(sample(dit, vae, encoder, [prompt], width=args.res, height=args.res,
                                     steps=args.steps, guidance=args.guidance, seed=args.seed + pi)[0])
        rows.append(panels)
        print(f"[render] {prompt[:40]} done", flush=True)

    w, h = rows[0][0].size
    strip = 22
    sheet = Image.new("RGB", (w * len(scales), (h + strip) * len(rows)), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for r, panels in enumerate(rows):
        y = r * (h + strip)
        d.text((4, y + 4), "scales: " + "  ".join(f"{s:+g}" for s in scales), fill=(0, 0, 0))
        for c, im in enumerate(panels):
            sheet.paste(im, (c * w, y + strip))
    sheet.save(args.out)
    print(f"[render] saved {args.out}", flush=True)
    print("SLIDER_RENDER_DONE", flush=True)


if __name__ == "__main__":
    main()
