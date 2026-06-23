#!/usr/bin/env python
"""Precompute the style-reference cache for Krea 2 (Qwen3-VL native image conditioning).

Feeds the **style image** through the Qwen3-VL VLM so image-derived conditioning enters the same
``(B, L, 12, 2560)`` stream the DiT already consumes. We cache that IMAGE-CONDITIONED ``llm_text``
alongside the target latent, in the **exact payload format of precache_t2i** -- so the existing DiT-only
trainer (``train_t2i_lora_cached.py`` / full-FT) trains the DiT/LoRA to *use* image conditioning with no
trainer change. Inference then uses ``sample.py --style-ref <image>``.

  python precache_style.py --config config/precache_t2i.yaml --manifest style.jsonl --data-root /data

Manifest line: {"target": "...", "style"|"ref"|"refs"[0]: "<style image>", "caption"|"text": "..."}
  - target : the image to reconstruct (its content + the reference style)
  - style  : a DIFFERENT image carrying the desired style (same-style/different-content pairs decouple
             style from content; ref==target degenerates to copying)
Payload: {"z_tgt","grid_h","grid_w","idx","caption","src","llm_text": image-conditioned (n,12,2560)}
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image

from constants import PATCH_SIZE
from loading import build_encoder, build_vae
from precache_t2i import ALIGN, atomic_save, encode_latent, images_to_tensor, patchify, round_to
from training_config import apply_runtime, dtype_of, load_config


def _caption(line: dict) -> str:
    for k in ("caption", "text", "instruction"):
        if line.get(k):
            return str(line[k])
    return ""


def _style_path(line: dict):
    if line.get("style"):
        return line["style"]
    if line.get("ref"):
        return line["ref"]
    if line.get("refs"):
        return line["refs"][0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", required=True, help="JSONL: one {target, style, caption} per line")
    ap.add_argument("--data-root", default=None, help="base dir for relative paths (default: manifest dir)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)
    res = cfg.data.resolution
    cache_dir = cfg.paths.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    root = args.data_root or os.path.dirname(os.path.abspath(args.manifest))

    with open(args.manifest, "r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    if args.limit:
        lines = lines[: args.limit]
    if not lines:
        raise SystemExit(f"no manifest lines in {args.manifest}")

    def resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(root, p)

    print(f"examples={len(lines)} shard={args.shard}/{args.num_shards} res={res} -> {cache_dir}", flush=True)
    vae = build_vae(cfg, device, dtype)
    encoder = build_encoder(cfg, device, dtype)  # frozen; used to image-condition the text features

    done = skipped = 0
    for idx, line in enumerate(lines):
        if idx % args.num_shards != args.shard:
            continue
        sp = _style_path(line)
        if "target" not in line or not sp:
            print(f"  WARN idx={idx}: missing target or style; skipping", flush=True)
            skipped += 1
            continue
        tgt, sty = resolve(line["target"]), resolve(sp)
        out = os.path.join(cache_dir, f"{idx:06d}.pt")
        if os.path.exists(out):
            try:
                if torch.load(out, map_location="cpu", weights_only=False).get("src") == os.path.abspath(tgt):
                    skipped += 1
                    continue
            except Exception:
                pass

        caption = _caption(line)
        try:
            img = Image.open(tgt)
            side = round_to(res, ALIGN)
            x = images_to_tensor(img, side, side, dtype).to(device)
            z = encode_latent(vae, x)
            z_tgt = patchify(z, PATCH_SIZE)[0].to(torch.bfloat16).cpu()  # (gh*gw, 64)
            gh = gw = side // ALIGN
            style_img = Image.open(sty).convert("RGB")
            hiddens, mask = encoder([caption], images=[style_img])       # image-conditioned (1,L,12,2560)
            llm_text = hiddens[0][mask[0]].to(torch.bfloat16).cpu()       # (n_text, 12, 2560)
        except Exception as e:
            print(f"  WARN encode failed idx={idx} ({tgt}): {type(e).__name__} {e}; skipping", flush=True)
            skipped += 1
            continue

        atomic_save({"z_tgt": z_tgt, "grid_h": gh, "grid_w": gw, "idx": idx, "caption": caption,
                     "src": os.path.abspath(tgt), "llm_text": llm_text}, out)
        done += 1
        if done % 25 == 0:
            print(f"  {done} cached ({skipped} skipped)", flush=True)

    if done == 0 and skipped == 0:
        raise SystemExit("0 cached -- check manifest / paths")
    print(f"DONE cached={done} skipped={skipped} -> {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
