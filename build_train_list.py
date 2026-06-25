#!/usr/bin/env python
"""Generate an oversampling ``train_list`` from an edit manifest, by ``edit_type`` family.

An edit corpus's natural distribution starves the hard edit classes (structural object ops and in-image
text are the rare tail), so a flat pass under-trains exactly the edits that lag. This buckets each
manifest line into a family and repeats its cache filename by a per-family multiplier, producing the
JSON list ``index_caches(train_list=...)`` consumes (repeats == oversampling).

IMPORTANT: the cache file for manifest line ``i`` is ``<i:06d>.pt`` (``precache_edit`` names caches by
manifest line order). So this MUST read the *same* manifest file that was/will be precached, and keys on
LINE ORDER — not on any index embedded in the image paths.

  python build_train_list.py --manifest /data/edits/edit_manifest_full.jsonl \
      --out /data/edits/train_list_balanced.json --w-structural 3 --w-text 4 --w-person 2 --w-global 1
"""
from __future__ import annotations

import argparse
import collections
import json
import sys

if hasattr(sys.stdout, "reconfigure"):       # edit_type strings carry unicode (↔, –); don't die on cp1251
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Distinctive lowercased substrings -> family. Checked in this ORDER (first hit wins); phrases are long
# enough to avoid traps ("scene context" contains "text"; global style types mention "anime"/"comic"; the
# Accessories type contains "remove"/"replace"). Unseen types fall through to "global".
RULES = [
    ("structural", ["remove an existing", "add a new object", "replace one object", "relocate",
                    "size/shape/orientation", "outpainting", "scene context", "attribute"]),
    ("text",       ["handwritten/printed", "in signs", "translate written", "font style"]),
    ("person",     ["of the person", "the person", "person to", "caricature", "age / gender",
                    "pose tweak", "clothing", "accessories", "modify express"]),
]


def family_of(edit_type: str) -> str:
    s = (edit_type or "").lower()
    for fam, kws in RULES:
        if any(k in s for k in kws):
            return fam
    return "global"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="the SAME jsonl fed to precache_edit")
    ap.add_argument("--out", required=True, help="output train_list JSON")
    ap.add_argument("--w-structural", type=int, default=3)
    ap.add_argument("--w-text", type=int, default=4)
    ap.add_argument("--w-person", type=int, default=2)
    ap.add_argument("--w-global", type=int, default=1)
    args = ap.parse_args()
    mult = {"structural": args.w_structural, "text": args.w_text,
            "person": args.w_person, "global": args.w_global}

    entries: list[str] = []
    by_type = collections.Counter()
    by_fam_in = collections.Counter()
    by_fam_out = collections.Counter()
    type_fam: dict[str, str] = {}

    with open(args.manifest, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln:
                continue
            et = json.loads(ln).get("edit_type", "")
            fam = family_of(et)
            type_fam[et] = fam
            by_type[et] += 1
            by_fam_in[fam] += 1
            reps = mult[fam]
            name = f"{i:06d}.pt"
            entries.extend([name] * reps)
            by_fam_out[fam] += reps

    with open(args.out, "w", encoding="utf-8") as g:
        json.dump(entries, g)

    n_in = sum(by_fam_in.values())
    n_out = len(entries)
    print(f"=== per-type classification ({len(by_type)} types) ===")
    for et, c in by_type.most_common():
        print(f"  [{type_fam[et]:>10}] x{mult[type_fam[et]]}  {c:7d}  {et}")
    print(f"\n=== family totals (in -> oversampled) ===")
    for fam in ("structural", "text", "person", "global"):
        pin = 100 * by_fam_in[fam] / n_in if n_in else 0
        pout = 100 * by_fam_out[fam] / n_out if n_out else 0
        print(f"  {fam:>10} x{mult[fam]}:  {by_fam_in[fam]:7d} ({pin:4.1f}%)  ->  {by_fam_out[fam]:7d} ({pout:4.1f}%)")
    print(f"\nDONE manifest_lines={n_in}  train_list_entries={n_out}  (epoch grows {n_out/n_in:.2f}x) -> {args.out}")


if __name__ == "__main__":
    main()
