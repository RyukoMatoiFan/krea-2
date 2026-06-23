#!/usr/bin/env python
"""Precompute the edit / multi-reference (and style-reference) training cache for Krea 2.

Each manifest line is a JSON object with a target image, one or more reference images, and a caption.
For each example we VAE-encode the target and every reference into normalized, patchified latent tokens
and (optionally) cache the Qwen3-VL text features. The edit trainer then packs
``[text, refs(clean), target(noised)]`` and supervises only the target (see ``edit_training_step``).
Style transfer uses the same path: ref = a style image, target = an image in that style.

  python precache_edit.py --config config/precache_t2i.yaml --manifest /data/edit/meta.jsonl \
      --data-root /data/edit

Manifest line (path keys relative to --data-root, or absolute):
  {"target": "train/000_tgt.jpg", "refs": ["train/000_ref0.jpg", ...],
   "caption"|"instruction"|"text": "..."}

Cache payload per example (``<idx:06d>.pt``):
  {"z_tgt": (n_tgt,64) bf16, "grid_h", "grid_w", "idx", "caption", "src",
   "refs": [{"tokens": (n_ref,64) bf16, "grid_h", "grid_w"}, ...],
   ["llm_text": (n_text,12,2560) bf16]}
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
    for k in ("caption", "instruction", "text"):
        if line.get(k):
            return str(line[k])
    return ""


def _refs_of(line: dict) -> list:
    """Accept ``refs: [...]`` (multiref) or a single ``source``/``src`` (single-ref edit)."""
    if line.get("refs"):
        return list(line["refs"])
    for k in ("source", "src", "ref"):
        if line.get(k):
            return [line[k]]
    return []


def _encode_image(vae, path: str, res: int, device, dtype):
    """RGB image -> (normalized, patchified) latent tokens + latent grid (gh, gw). Square-resized."""
    img = Image.open(path)
    side = round_to(res, ALIGN)
    x = images_to_tensor(img, side, side, dtype).to(device)
    z = encode_latent(vae, x)                                     # (1,16,H_lat,W_lat)
    tokens = patchify(z, PATCH_SIZE)[0].to(torch.bfloat16).cpu()  # (gh*gw, 64)
    return tokens, side // ALIGN, side // ALIGN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", required=True, help="JSONL: one {target, refs[], caption} per line")
    ap.add_argument("--data-root", default=None, help="base dir for relative paths (default: manifest dir)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--cache-text", dest="cache_text", action="store_true", default=True)
    ap.add_argument("--no-cache-text", dest="cache_text", action="store_false")
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

    print(f"examples={len(lines)} shard={args.shard}/{args.num_shards} res={res} "
          f"cache_text={args.cache_text} -> {cache_dir}", flush=True)
    vae = build_vae(cfg, device, dtype)
    encoder = build_encoder(cfg, device, dtype) if args.cache_text else None

    done = skipped = 0
    for idx, line in enumerate(lines):
        if idx % args.num_shards != args.shard:
            continue
        refs = [resolve(r) for r in _refs_of(line)]
        if "target" not in line or not refs:
            print(f"  WARN idx={idx}: missing target or refs; skipping", flush=True)
            skipped += 1
            continue
        tgt = resolve(line["target"])
        out = os.path.join(cache_dir, f"{idx:06d}.pt")
        if os.path.exists(out):  # content-aware skip
            try:
                if torch.load(out, map_location="cpu", weights_only=False).get("src") == os.path.abspath(tgt):
                    skipped += 1
                    continue
            except Exception:
                pass

        caption = _caption(line)
        try:
            z_tgt, gh, gw = _encode_image(vae, tgt, res, device, dtype)
            ref_payload = []
            for r in refs:
                rt, rgh, rgw = _encode_image(vae, r, res, device, dtype)
                ref_payload.append({"tokens": rt, "grid_h": rgh, "grid_w": rgw})
        except Exception as e:
            print(f"  WARN encode failed idx={idx} ({tgt}): {type(e).__name__} {e}; skipping", flush=True)
            skipped += 1
            continue

        payload = {"z_tgt": z_tgt, "grid_h": gh, "grid_w": gw, "idx": idx,
                   "caption": caption, "src": os.path.abspath(tgt), "refs": ref_payload}
        if args.cache_text:
            hiddens, mask = encoder([caption])
            payload["llm_text"] = hiddens[0][mask[0]].to(torch.bfloat16).cpu()
        atomic_save(payload, out)
        done += 1
        if done % 25 == 0:
            print(f"  {done} cached ({skipped} skipped)", flush=True)

    if done == 0 and skipped == 0:
        raise SystemExit("0 cached -- check manifest / paths")
    print(f"DONE cached={done} skipped={skipped} -> {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
