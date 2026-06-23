#!/usr/bin/env python
"""Precompute the per-image training cache for Krea 2 text-to-image fine-tuning.

Loads ONLY the encoders (AutoencoderKLQwenImage VAE + frozen Qwen3-VL-4B), NOT the 12B DiT, then for
each image writes one ``<idx:06d>.pt`` holding the normalized, patchified latent tokens + the caption
(+ optionally the cached text features). Training is then encoder-free (the joint trainer re-encodes
captions live through the trainable TE; the DiT-only trainer reuses the cached ``llm_text``).

  python precache_t2i.py --config config/precache_t2i.yaml
  python precache_t2i.py --config ... --num-shards 3 --shard 0   # one process per GPU

Cache payload per image:
  {"z_tgt": (gh*gw, 64) bf16, "grid_h": gh, "grid_w": gw, "idx": int,
   "caption": str, "src": <abs source path>, ["llm_text": (n_text, 12, 2560) bf16]}
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from constants import PATCH_SIZE, VAE_COMPRESSION
from loading import build_encoder, build_vae
from training_config import apply_runtime, dtype_of, load_config
from training_utils import aspect_buckets, nearest_bucket

ALIGN = VAE_COMPRESSION * PATCH_SIZE  # image side must be a multiple of this (=16)


def round_to(v: int, m: int) -> int:
    return max(m, ((v + m - 1) // m) * m)


def images_to_tensor(img: Image.Image, width: int, height: int, dtype) -> torch.Tensor:
    """RGB PIL -> (1, 3, H, W) in [-1, 1]."""
    img = img.convert("RGB").resize((width, height))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(dtype=dtype)


@torch.no_grad()
def encode_latent(vae, images: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [-1,1] -> normalized latent (B,16,H_lat,W_lat) via AutoencoderKLQwenImage."""
    x = rearrange(images, "b c h w -> b c 1 h w")  # QwenImage VAE is a (B,C,T,H,W) video VAE
    z = vae.ae.encode(x).latent_dist.mode()        # (B,16,1,H_lat,W_lat)
    z = (z - vae.latents_mean) / vae.latents_std
    return rearrange(z, "b c 1 h w -> b c h w")


def patchify(z: torch.Tensor, patch: int) -> torch.Tensor:
    """(B,16,H_lat,W_lat) -> (B, gh*gw, 16*patch*patch), matching mmdit.first / sampling.prepare."""
    return rearrange(z, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)


def read_caption(image_path: str, cfg) -> str | None:
    """Read the caption for an image: verbatim JSON sidecar (prebuilt_json) or a .txt file."""
    if cfg.data.prebuilt_json:
        jp = os.path.splitext(image_path)[0] + cfg.data.json_suffix
        if not os.path.exists(jp):
            return None
        with open(jp, "r", encoding="utf-8") as f:
            return f.read().strip()
    tp = os.path.splitext(image_path)[0] + ".txt"
    if os.path.exists(tp):
        with open(tp, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def atomic_save(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--img-dir", default=None, help="override the image folder (default: paths.data_root)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--cache-text", dest="cache_text", action="store_true", default=True,
                    help="cache Qwen3-VL text features (DiT-only trainer); default on")
    ap.add_argument("--no-cache-text", dest="cache_text", action="store_false",
                    help="skip text features (joint trainer re-encodes live; big runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)

    img_dir = args.img_dir or cfg.paths.data_root
    cache_dir = cfg.paths.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    images = sorted(glob.glob(os.path.join(img_dir, f"*.{cfg.data.img_ext}")))
    if not images:
        raise SystemExit(f"no *.{cfg.data.img_ext} images under {img_dir}")
    if args.limit:
        images = images[: args.limit]

    # Persistent relpath->idx map so re-runs keep stable indices (eval holdout = idx < n_eval_holdout).
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    for p in images:
        rel = os.path.relpath(p, img_dir)
        if rel not in manifest:
            manifest[rel] = len(manifest)
    if args.shard == 0:
        atomic_save_json(manifest, manifest_path)

    # Source guard: refuse to mix two different sources into one cache dir.
    src_info = {"img_dir": os.path.abspath(img_dir), "prebuilt_json": cfg.data.prebuilt_json}
    src_path = os.path.join(cache_dir, "precache_source.json")
    if os.path.exists(src_path):
        with open(src_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev != src_info:
            raise SystemExit(f"cache_dir {cache_dir} was built from {prev}, not {src_info}")
    elif args.shard == 0:
        atomic_save_json(src_info, src_path)

    buckets = None
    if cfg.data.aspect_bucketing:
        buckets = aspect_buckets(cfg.data.bucket_pixels or cfg.data.resolution ** 2, divisor=ALIGN,
                                 num=cfg.data.num_buckets)

    print(f"images={len(images)} shard={args.shard}/{args.num_shards} cache_text={args.cache_text} "
          f"-> {cache_dir}", flush=True)
    vae = build_vae(cfg, device, dtype)
    encoder = build_encoder(cfg, device, dtype) if args.cache_text else None

    done = skipped = 0
    for p in images:
        rel = os.path.relpath(p, img_dir)
        idx = manifest[rel]
        if idx % args.num_shards != args.shard:
            continue
        out = os.path.join(cache_dir, f"{idx:06d}.pt")
        src = os.path.abspath(p)
        if os.path.exists(out):  # content-aware skip
            try:
                prev = torch.load(out, map_location="cpu", weights_only=False)
                if prev.get("src") == src:
                    skipped += 1
                    continue
            except Exception:
                pass

        caption = read_caption(p, cfg)
        if caption is None:
            print(f"  WARN no caption sidecar for {rel}; skipping", flush=True)
            skipped += 1
            continue

        try:
            img = Image.open(p)
            if buckets is not None:
                H, W = nearest_bucket(img.width, img.height, buckets)
            else:
                H = W = round_to(cfg.data.resolution, ALIGN)
            x = images_to_tensor(img, W, H, dtype).to(device)
            z = encode_latent(vae, x)                       # (1,16,H_lat,W_lat)
            tokens = patchify(z, PATCH_SIZE)[0].to(torch.bfloat16).cpu()  # (gh*gw, 64)
            gh, gw = H // ALIGN, W // ALIGN
        except Exception as e:
            print(f"  WARN encode failed for {rel}: {type(e).__name__} {e}; skipping", flush=True)
            skipped += 1
            continue

        payload = {"z_tgt": tokens, "grid_h": gh, "grid_w": gw, "idx": idx,
                   "caption": caption, "src": src}
        if args.cache_text:
            hiddens, mask = encoder([caption])              # (1,L,12,2560), (1,L) bool
            payload["llm_text"] = hiddens[0][mask[0]].to(torch.bfloat16).cpu()  # (n_text,12,2560)
        atomic_save(payload, out)
        done += 1
        if done % 50 == 0:
            print(f"  {done} cached ({skipped} skipped)", flush=True)

    if done == 0 and skipped == 0:
        raise SystemExit("shard produced 0 caches -- check img_dir / captions")
    print(f"DONE cached={done} skipped={skipped} -> {cache_dir}", flush=True)


def atomic_save_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
