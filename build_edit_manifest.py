#!/usr/bin/env python
"""Build the full edit-training manifest from a source SFT ``jsonl``.

Each source line carries an instruction (``--instr-field``, default ``text``) and an ``edit_type``. The
downloaded pairs live in ``--train-dir`` as ``<i:06d>_src.jpg`` / ``<i:06d>_tgt.jpg`` indexed by the line
number ``i``. This emits one
``precache_edit``-shaped record per line::

    {"target": "train/000123_tgt.jpg", "refs": ["train/000123_src.jpg"],
     "instruction": "<text>", "edit_type": "<edit_type>"}

Index ``i`` is preserved as the line order, so cache file ``<i:06d>.pt`` (named by ``precache_edit``)
maps back to the same ``edit_type`` — which ``build_train_list.py`` relies on for oversampling.

  python build_edit_manifest.py --sft /data/edits/sft.jsonl --train-dir /data/edits/train \
      --out /data/edits/edit_manifest_full.jsonl --require-images
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="source sft.jsonl")
    ap.add_argument("--train-dir", default="train",
                    help="dir holding <i:06d>_src.jpg/_tgt.jpg, as referenced from the manifest (default: train)")
    ap.add_argument("--abs-train-dir", default=None,
                    help="filesystem dir to existence-check against (default: --train-dir)")
    ap.add_argument("--out", required=True, help="output manifest jsonl")
    ap.add_argument("--instr-field", default="text", help="sft.jsonl field holding the instruction")
    ap.add_argument("--limit", type=int, default=0, help="only the first N lines (0 = all)")
    ap.add_argument("--require-images", action="store_true",
                    help="skip a line if its _src/_tgt jpg is missing on disk")
    args = ap.parse_args()

    check_dir = args.abs_train_dir or args.train_dir
    rel = args.train_dir
    written = skipped_missing = skipped_blank = 0
    with open(args.sft, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            if args.limit and written + skipped_missing + skipped_blank >= args.limit:
                break
            d = json.loads(ln)
            instr = (d.get(args.instr_field) or "").strip()
            if not instr:
                skipped_blank += 1
                continue
            src = f"{i:06d}_src.jpg"
            tgt = f"{i:06d}_tgt.jpg"
            if args.require_images and not (
                os.path.exists(os.path.join(check_dir, src)) and os.path.exists(os.path.join(check_dir, tgt))
            ):
                skipped_missing += 1
                continue
            rec = {
                "target": f"{rel}/{tgt}",
                "refs": [f"{rel}/{src}"],
                "instruction": instr,
                "edit_type": d.get("edit_type", ""),
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % 25000 == 0:
                print(f"  {written} written...", flush=True)

    print(f"DONE wrote={written} skipped_missing={skipped_missing} skipped_blank={skipped_blank} -> {args.out}")
    if written == 0:
        raise SystemExit("0 records written -- check --sft / --train-dir / --instr-field")


if __name__ == "__main__":
    main()
