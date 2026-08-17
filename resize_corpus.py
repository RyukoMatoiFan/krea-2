#!/usr/bin/env python
"""Build a transfer-sized copy of an image corpus: short edge to ``--short``, re-encoded as WebP.

Source images are typically far larger than the training resolution, so shipping them unchanged
spends transfer time on pixels that precache discards. Downscaling first makes the copy a small
fraction of the original.

Ships pixels rather than VAE latents on purpose: at this resolution a re-encoded image is no larger
than its latent, and keeping pixels leaves the bucketing, resolution and captioning decisions open
instead of freezing them into the cache.

Resumable by design -- an existing output is skipped, so an interrupted run is restarted, not
repeated. Reads ids in list order, so ``--limit`` on a shuffled list yields a uniform random subset
rather than a biased prefix.

    python resize_corpus.py --list ids.csv --src /data/originals --out /data/shipped --limit 150000
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil

from PIL import Image

# Source corpora can include very large files; Pillow's decompression-bomb guard is aimed at
# malicious input and trips on legitimate high-resolution images from a corpus we control and have
# already filtered by resolution.
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = ("jpg", "jpeg", "png", "webp", "gif")


def shard_dir(root: str, image_id: int) -> str:
    """Id-sharded layout, 1000 ids per directory, so no folder is slow to list. Input and output
    use the same scheme, so the source corpus must already be laid out this way."""
    return os.path.join(root, "orig", f"{image_id // 1000:05d}")


def find_source(src: str, image_id: int) -> str | None:
    d = shard_dir(src, image_id)
    for e in IMAGE_EXTS:
        p = os.path.join(d, f"{image_id}.{e}")
        if os.path.exists(p):
            return p
    return None


def process(job) -> tuple:
    image_id, src, out, short, long_cap, quality = job
    od = shard_dir(out, image_id)
    op = os.path.join(od, f"{image_id}.webp")
    tp = os.path.join(od, f"{image_id}.txt")
    # Decide "already done" BEFORE touching the source. Probing the source first costs up to five
    # stat calls on the input volume for every id, including the ones already finished. When the
    # input is slower than the output volume that dominates a resumed pass. Order matters more than
    # the work here.
    if os.path.exists(op) and os.path.exists(tp):
        return ("skip", 0, 0, 0)
    sp = find_source(src, image_id)
    if sp is None:
        return ("missing", 0, 0, 0)
    src_txt = os.path.splitext(sp)[0] + ".txt"
    os.makedirs(od, exist_ok=True)
    try:
        with Image.open(sp) as im:
            w0, h0 = im.size
            s0 = short / min(w0, h0)
            if max(w0, h0) * s0 > long_cap:
                s0 = long_cap / max(w0, h0)
            if s0 < 1.0 and im.format == "JPEG":
                # Decode straight to a reduced DCT scale instead of unpacking full resolution and
                # throwing most of it away. No-op for other formats, and it only ever lands on a size
                # at or above the target, so the LANCZOS step below still does the exact resize.
                im.draft("RGB", (max(1, round(w0 * s0)), max(1, round(h0 * s0))))
            # Flatten to RGB: alpha and palette modes are not what the VAE consumes, and deciding the
            # matte here (rather than at precache) keeps the shipped copy self-describing.
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            scale = short / min(w, h)
            capped = 0
            if max(w, h) * scale > long_cap:
                # Extreme aspect ratios would otherwise ship a very long image whose short edge is
                # already at target. Capping the long edge bounds the file, at the cost of a short
                # edge below target for those few. Counted and reported rather than done silently.
                scale = long_cap / max(w, h)
                capped = 1
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.LANCZOS)
            im.save(op, "WEBP", quality=quality, method=4)
    except Exception as e:
        return ("error:" + type(e).__name__, 0, 0, 0)
    if os.path.exists(src_txt):
        shutil.copyfile(src_txt, tp)
    return ("ok", os.path.getsize(sp), os.path.getsize(op), capped)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="csv with an 'id' column, in the order to process")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--short", type=int, default=1024, help="target short edge")
    ap.add_argument("--long-cap", type=int, default=2048)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0, help="first N ids of the list; 0 = all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    with open(args.list, newline="", encoding="utf-8") as fh:
        ids = [int(r["id"]) for r in csv.DictReader(fh)]
    if args.limit:
        ids = ids[: args.limit]
    # Select in LIST order (a shuffled list makes --limit a uniform sample), then sort for disk
    # locality. The two are in direct tension on spinning media: the same shuffle that makes a prefix
    # unbiased turns the read pattern into random seeks across the whole corpus, which is ruinous on
    # rotational media. Sorting restores a sequential directory walk without changing which ids were
    # chosen.
    ids.sort()
    jobs = [(i, args.src, args.out, args.short, args.long_cap, args.quality) for i in ids]
    print(f"to process: {len(jobs):,} with {args.workers} workers", flush=True)

    stats = {"ok": 0, "skip": 0, "missing": 0}
    bin_ = bout = capped = errors = 0
    with mp.Pool(args.workers) as pool:
        for n, (status, si, so, cap) in enumerate(
                pool.imap_unordered(process, jobs, chunksize=64), 1):
            if status.startswith("error"):
                errors += 1
                stats[status] = stats.get(status, 0) + 1
            else:
                stats[status] = stats.get(status, 0) + 1
            bin_ += si
            bout += so
            capped += cap
            if n % 20000 == 0:
                ratio = (bout / bin_) if bin_ else 0
                print(f"{n:,}/{len(jobs):,}  ok {stats['ok']:,}  skip {stats['skip']:,}  "
                      f"missing {stats['missing']:,}  err {errors:,}  "
                      f"{bout/1e9:.1f} GB out ({ratio:.2f}x)", flush=True)
    ratio = (bout / bin_) if bin_ else 0
    print(f"done: {stats}  long-edge-capped {capped:,}")
    print(f"bytes: {bin_/1e9:.1f} GB -> {bout/1e9:.1f} GB  ({ratio:.3f}x)")


if __name__ == "__main__":
    main()
