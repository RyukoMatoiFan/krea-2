#!/usr/bin/env python
"""Bring a captioned retention set into the training corpus format.

The retention slice exists to stop a single-domain fine-tune from destroying what the base model
already does -- photorealism and text rendering are the first capabilities to go. It arrives as
source-resolution images plus a JSONL manifest, and has to end up looking exactly like the rest of
the corpus: 1024px WebP with a ``.txt`` caption sidecar, sharded so no directory is slow to list.

Resolution and format match ``resize_corpus.py`` so a single precache pass covers everything with
one ``img_ext``.

    python prepare_retention.py --meta <set>/train/metadata.jsonl \\
        --root <set>/train --out images/retention --prefix <short-tag>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def process(job):
    rel, caption, root, out, prefix, short, long_cap, quality, shard_size = job
    src = os.path.join(root, rel)
    stem = prefix + "_" + os.path.splitext(os.path.basename(rel))[0]
    # Hash the name into shards rather than using a running counter: stable across reruns and
    # partial sets, so a resumed pass writes every file where it would have gone. hashlib, NOT the
    # builtin hash(): string hashing is salted per interpreter, so builtin hash would scatter the
    # same file to a different shard in every worker process and on every run.
    shard = "%05d" % (int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16) % shard_size)
    d = os.path.join(out, shard)
    op, tp = os.path.join(d, stem + ".webp"), os.path.join(d, stem + ".txt")
    if os.path.exists(op) and os.path.exists(tp):
        return ("skip", 0, 0)
    if not os.path.exists(src):
        return ("missing", 0, 0)
    os.makedirs(d, exist_ok=True)
    try:
        with Image.open(src) as im:
            w, h = im.size
            scale = short / min(w, h)
            if max(w, h) * scale > long_cap:
                scale = long_cap / max(w, h)
            if scale < 1.0 and im.format == "JPEG":
                im.draft("RGB", (max(1, round(w * scale)), max(1, round(h * scale))))
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            scale = short / min(w, h)
            if max(w, h) * scale > long_cap:
                scale = long_cap / max(w, h)
            if scale < 1.0:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            im.save(op, "WEBP", quality=quality, method=4)
    except Exception as e:
        return ("error:" + type(e).__name__, 0, 0)
    with io.open(tp, "w", encoding="utf-8", newline="\n") as f:
        f.write(" ".join(caption.split()).strip() + "\n")
    return ("ok", os.path.getsize(src), os.path.getsize(op))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="jsonl with file_name and text")
    ap.add_argument("--root", required=True, help="directory file_name is relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", required=True, help="prepended to every stem, keeps sources apart")
    ap.add_argument("--short", type=int, default=1024)
    ap.add_argument("--long-cap", type=int, default=2048)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--file-key", default="file_name")
    args = ap.parse_args()

    jobs = []
    with io.open(args.meta, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            rel, cap = rec.get(args.file_key), rec.get(args.text_key)
            if rel and cap:
                jobs.append((rel, cap, args.root, args.out, args.prefix, args.short,
                             args.long_cap, args.quality, args.shards))
    print(f"to process: {len(jobs):,} with {args.workers} workers", flush=True)

    stats, bin_, bout = {}, 0, 0
    with mp.Pool(args.workers) as pool:
        for n, (status, si, so) in enumerate(pool.imap_unordered(process, jobs, chunksize=32), 1):
            stats[status] = stats.get(status, 0) + 1
            bin_ += si
            bout += so
            if n % 2000 == 0:
                print(f"{n:,}/{len(jobs):,}  {stats}  {bout/1e9:.1f} GB out", flush=True)
    print(f"done: {stats}")
    print(f"bytes: {bin_/1e9:.1f} GB -> {bout/1e9:.1f} GB")


if __name__ == "__main__":
    main()
