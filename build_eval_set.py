#!/usr/bin/env python
"""Carve a held-out per-family eval set out of an edit manifest, so success rate is *measured*.

Picks ``--per-type`` examples from the TAIL of each ``edit_type`` (stable + trivial to exclude from
training) and emits, under ``--out-dir``:
  - ``eval_holdout.jsonl``        : every picked {idx,src,tgt,instruction,edit_type} (abs paths)
  - ``eval_preview.jsonl``        : a small balanced subset (``--preview-per-family`` each) for the
                                    trainer's EDIT_PREVIEW_MANIFEST -> [src|edit|tgt] rows every ckpt
  - ``eval_holdout_indices.json`` : the manifest LINE indices, for ``build_train_list.py --exclude``

Keep the preview small (it renders every checkpoint); the full holdout is for periodic deeper eval.

  python build_eval_set.py --manifest /data/edits/edit_manifest_full.jsonl --data-root /data/edits \
      --per-type 8 --preview-per-family 4 --out-dir /data/edits/eval
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_train_list import family_of   # reuse the same edit_type -> family rules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="the SAME jsonl fed to precache_edit / build_train_list")
    ap.add_argument("--data-root", required=True, help="prefix for the manifest's relative src/tgt paths")
    ap.add_argument("--per-type", type=int, default=8, help="held-out examples per edit_type")
    ap.add_argument("--preview-per-family", type=int, default=4, help="rendered-every-ckpt previews per family")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    by_type: dict[str, list] = collections.defaultdict(list)
    with open(args.manifest, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            by_type[json.loads(ln).get("edit_type", "")].append((i, json.loads(ln)))

    def absp(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(args.data_root, p)

    def entry(i: int, d: dict) -> dict:
        return {"idx": i, "src": absp(d["refs"][0]), "tgt": absp(d["target"]),
                "instruction": d["instruction"], "edit_type": d.get("edit_type", "")}

    holdout = [entry(i, d) for items in by_type.values() for i, d in items[-args.per_type:]]

    by_fam: dict[str, list] = collections.defaultdict(list)
    for e in holdout:
        by_fam[family_of(e["edit_type"])].append(e)
    preview = [e for es in by_fam.values() for e in es[: args.preview_per_family]]

    indices = sorted(e["idx"] for e in holdout)
    with open(os.path.join(args.out_dir, "eval_holdout.jsonl"), "w", encoding="utf-8") as g:
        for e in holdout:
            g.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "eval_preview.jsonl"), "w", encoding="utf-8") as g:
        for e in preview:
            g.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "eval_holdout_indices.json"), "w") as g:
        json.dump(indices, g)

    print(f"holdout={len(holdout)} ({len(by_type)} types x up to {args.per_type})  preview={len(preview)}  -> {args.out_dir}")
    print("preview per family:", {f: len(es[: args.preview_per_family]) for f, es in by_fam.items()})


if __name__ == "__main__":
    main()
