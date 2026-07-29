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
   "caption"|"instruction"|"text": "...",
   "reference_delta_mask": true, "validation_group": "optional-group"}

Cache payload per example (``<idx:06d>.pt``):
  {"z_tgt": (n_tgt,64) bf16, "grid_h", "grid_w", "idx", "caption", "src",
   "refs": [{"tokens": (n_ref,64) bf16, "grid_h", "grid_w"}, ...], "ref_paths": [abs src path, ...],
   ["reference_delta_mask": bool], ["validation_group": str],
   ["llm_text": (n_text,12,2560) bf16]}
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from PIL import Image

from constants import PATCH_SIZE
from edit_geometry import prepare_reference_image
from loading import build_encoder, build_vae
from precache_t2i import ALIGN, atomic_save, encode_latent, images_to_tensor, patchify, round_to
from training_config import apply_runtime, dtype_of, load_config
from training_utils import aspect_buckets, nearest_bucket

_GROUP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _stat_id(path: str) -> str:
    """size+mtime as a content proxy. A full content hash over the whole dataset costs more than the
    encode it protects; size+mtime catches an edited or replaced file, which is the real risk."""
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return "missing"


def _payload_sig(
    tgt: str,
    refs: list,
    caption: str,
    res: int,
    cache_text: bool,
    *,
    annotations: dict | None = None,
    system_prompt: str = "",
    encoder_id: str = "",
    vae_id: str = "",
    aspect_bucketing: bool = False,
    bucket_pixels: int = 0,
    num_buckets: int = 9,
    edit_ref_geometry: str = "square",
    fit_ref_crop_tolerance: float = 0.08,
) -> str:
    """Fingerprint of everything a cached payload depends on.

    Reference ORDER is significant (each reference occupies its own RoPE frame, so swapping two
    references is a different training example), which is why this hashes the ordered list rather
    than a set.

    Includes the PRODUCERS of the payload, not just its inputs: the text encoder identity and system
    prompt (they determine cached text), the VAE identity (it determines every latent), and file
    size+mtime (an image can be replaced in place). Hashing only path strings would not detect
    changes to producer settings when every input path stays identical.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(os.path.abspath(tgt).encode("utf-8"))
    h.update(_stat_id(tgt).encode("utf-8"))
    for r in refs:                       # ordered on purpose
        h.update(b"\x00")
        h.update(os.path.abspath(r).encode("utf-8"))
        h.update(_stat_id(r).encode("utf-8"))
    h.update(b"\x01")
    h.update((caption or "").encode("utf-8"))
    h.update(
        f"|res={int(res)}|text={bool(cache_text)}|aspect={bool(aspect_bucketing)}"
        f"|bucket_pixels={int(bucket_pixels)}|num_buckets={int(num_buckets)}"
        f"|ref_geometry={edit_ref_geometry}|fit_tol={float(fit_ref_crop_tolerance):.8g}"
        .encode("utf-8"))
    h.update(f"|sys={system_prompt}|enc={encoder_id}|vae={vae_id}".encode("utf-8"))
    if annotations:
        h.update(b"|annotations=")
        h.update(json.dumps(annotations, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()[:32]


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


def _annotations_of(line: dict) -> dict:
    """Validate and copy optional trainer annotations from a manifest row."""
    out = {}
    if "reference_delta_mask" in line:
        value = line["reference_delta_mask"]
        if not isinstance(value, bool):
            raise ValueError("reference_delta_mask must be a JSON boolean")
        out["reference_delta_mask"] = value
    if "validation_group" in line:
        value = line["validation_group"]
        if not isinstance(value, str) or not _GROUP_RE.fullmatch(value):
            raise ValueError(f"validation_group must match {_GROUP_RE.pattern!r}")
        out["validation_group"] = value
    return out


def _encode_image(vae, image: Image.Image, width: int, height: int, device, dtype):
    """RGB image -> normalized, patchified latent tokens and its latent grid."""
    x = images_to_tensor(image, width, height, dtype).to(device)
    z = encode_latent(vae, x)                                     # (1,16,H_lat,W_lat)
    tokens = patchify(z, PATCH_SIZE)[0].to(torch.bfloat16).cpu()  # (gh*gw, 64)
    return tokens, height // ALIGN, width // ALIGN


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
    buckets = None
    if cfg.data.aspect_bucketing:
        buckets = aspect_buckets(
            cfg.data.bucket_pixels or res ** 2, divisor=ALIGN, num=cfg.data.num_buckets)
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
          f"buckets={bool(buckets)} ref_geometry={cfg.data.edit_ref_geometry} "
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
        caption = _caption(line)
        try:
            annotations = _annotations_of(line)
        except ValueError as e:
            print(f"  WARN idx={idx}: {e}; skipping", flush=True)
            skipped += 1
            continue
        # The payload depends on the target, the ORDERED references, the caption and the
        # resolution -- validating only the target path lets a changed reference order, an added
        # reference, an edited instruction or a different resolution silently reuse a stale cache.
        sig = _payload_sig(tgt, refs, caption, res, args.cache_text,
                           annotations=annotations,
                           system_prompt=getattr(cfg.data, "text_system_prompt", ""),
                           encoder_id=cfg.paths.text_encoder_id, vae_id=cfg.paths.vae_id,
                           aspect_bucketing=cfg.data.aspect_bucketing,
                           bucket_pixels=cfg.data.bucket_pixels,
                           num_buckets=cfg.data.num_buckets,
                           edit_ref_geometry=cfg.data.edit_ref_geometry,
                           fit_ref_crop_tolerance=cfg.data.fit_ref_crop_tolerance)
        out = os.path.join(cache_dir, f"{idx:06d}.pt")
        if os.path.exists(out):  # content-aware skip
            try:
                prev = torch.load(out, map_location="cpu", weights_only=False)
                if prev.get("sig") == sig:
                    skipped += 1
                    continue
                # A signature-less cache cannot be shown to match the current caption, cache_text,
                # system prompt, encoder or VAE, so it is recached rather than trusted.
                why = "no signature" if prev.get("sig") is None \
                    else "signature mismatch"
                print(f"  RECACHE idx={idx}: {why}", flush=True)
            except Exception:
                pass

        try:
            target_image = Image.open(tgt).convert("RGB")
            if buckets is not None:
                target_h, target_w = nearest_bucket(
                    target_image.width, target_image.height, buckets)
            else:
                target_h = target_w = round_to(res, ALIGN)
            z_tgt, gh, gw = _encode_image(
                vae, target_image, target_w, target_h, device, dtype)
            ref_payload = []
            for r in refs:
                ref_image = prepare_reference_image(
                    Image.open(r),
                    (target_w, target_h),
                    geometry=cfg.data.edit_ref_geometry,
                    align=ALIGN,
                    crop_tolerance=cfg.data.fit_ref_crop_tolerance,
                )
                rt, rgh, rgw = _encode_image(
                    vae, ref_image, ref_image.width, ref_image.height, device, dtype)
                ref_payload.append({"tokens": rt, "grid_h": rgh, "grid_w": rgw})
        except Exception as e:
            print(f"  WARN encode failed idx={idx} ({tgt}): {type(e).__name__} {e}; skipping", flush=True)
            skipped += 1
            continue

        payload = {"z_tgt": z_tgt, "grid_h": gh, "grid_w": gw, "idx": idx,
                   "caption": caption, "src": os.path.abspath(tgt), "refs": ref_payload,
                   "ref_paths": [os.path.abspath(r) for r in refs],  # source pixels for optional VLM cond
                   "edit_ref_geometry": cfg.data.edit_ref_geometry,
                   "sig": sig}  # fingerprint of everything the payload depends on (see the skip check)
        payload.update(annotations)
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
