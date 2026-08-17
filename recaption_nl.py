#!/usr/bin/env python
"""Rewrite a deterministic slice of the corpus with natural-language captions.

A tag-captioned fine-tune plus a natural-language retention slice makes caption style perfectly
correlated with domain: prose always means the retention set, tags always mean the target set. The
model can satisfy both by learning that correlation, and then a natural-language request for the
fine-tuned domain lands in a hole neither slice covers -- a capability the base model has and the
fine-tune would lose. Recaptioning a fraction of the corpus in prose breaks the correlation.

The slice is chosen by hashing the id, so it is uniform, reproducible, and needs no ordering file.
The original tag caption is preserved beside the new one (``.tags.txt``), which makes the pass
reversible and doubles as the resume marker.

Captions are baked into the cache at precache time, so this must run BEFORE precache:
``precache_t2i.py`` skips an image whose cached ``src`` path matches and would keep a stale caption.

Two backends. ``ollama`` is the default: a serving runtime keeps the model resident across requests
and sequences them itself, which sidesteps the batched-decoder padding trap described below. The
``transformers`` backend needs no server but batches, so it carries that trap.

    python recaption_nl.py --images <corpus> --fraction 10 \\
        --backend ollama --model <vlm-tag> --url http://127.0.0.1:11434 --workers 8
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

# Domain-neutral by default. A corpus with a specific look is better served by --prompt, which
# keeps the wording with the run recipe instead of hard-coding one domain into the tool.
PROMPT = (
    "Describe this image in one or two natural sentences, as a person would describe it to "
    "someone who cannot see it. Cover the subject, what they are doing, the setting and the mood. "
    "Write flowing prose, not a list of tags or keywords. Do not begin with 'This image shows'."
)


def selected(image_id: str, fraction: int) -> bool:
    """Deterministic 1-in-`fraction` slice. Hashing rather than ``id % fraction`` because ids are
    often assigned sequentially, so low digits can correlate with when an image was added."""
    return int(hashlib.sha1(image_id.encode()).hexdigest()[:8], 16) % fraction == 0


def as_jpeg_b64(path: str, side: int) -> str:
    """Ollama refuses WebP ("Failed to load image or audio file") even for a vision-capable model,
    so the corpus format has to be transcoded on the way in. The captioner gains nothing from full
    resolution and attention is quadratic in image tokens."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def strip_thinking(text: str) -> str:
    """Drop any reasoning the model emits before its answer.

    ``think: false`` suppresses the reasoning *content* but the closing marker still appears at the
    start of the reply, so captions arrive as ``</think> <the actual caption> ...``. Cutting
    at the last marker also handles a model that ignores the flag and emits the whole block.
    """
    for marker in ("</think>", "</thinking>"):
        if marker in text:
            text = text.rsplit(marker, 1)[1]
    return text.strip()


def write_caption(path: str, caption: str) -> bool:
    caption = " ".join(strip_thinking(caption).split()).strip()
    if not caption:
        return False
    stem = os.path.splitext(path)[0]
    tags_path, txt_path = stem + ".tags.txt", stem + ".txt"
    # Back the tag caption up FIRST: it is the only copy, so writing the backup before the
    # replacement means an interrupted run can never destroy one.
    if os.path.exists(txt_path) and not os.path.exists(tags_path):
        with io.open(txt_path, encoding="utf-8") as f:
            original = f.read()
        with io.open(tags_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(original)
    with io.open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(caption + "\n")
    return True


def caption_ollama(path: str, args) -> str:
    payload = {
        "model": args.model,
        "stream": False,
        # Reasoning models left in thinking mode spend the token budget on hidden reasoning and
        # return a truncated caption; captioning needs none of it.
        "think": False,
        "messages": [{"role": "user", "content": args.prompt,
                      "images": [as_jpeg_b64(path, args.side)]}],
        "options": {"num_predict": args.max_new_tokens},
    }
    req = urllib.request.Request(args.url.rstrip("/") + "/api/chat",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        d = json.load(r)
    return ((d.get("message") or {}).get("content") or "").strip()


def run_ollama(todo, args):
    done = [0]
    t0 = time.time()

    def work(p):
        try:
            cap = caption_ollama(p, args)
        except Exception as e:
            print(f"  WARN {os.path.basename(p)}: {type(e).__name__} {str(e)[:80]}", flush=True)
            return
        if write_caption(p, cap):
            done[0] += 1
            n = done[0]
            if n % 200 == 0:
                el = time.time() - t0
                rate = n / el
                print(f"{n:,}/{len(todo):,}  {rate:.2f} img/s  "
                      f"eta {(len(todo) - n) / max(rate, 1e-9) / 3600:.1f} h", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    return done[0], time.time() - t0


def run_transformers(todo, args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    # Left-pad or every caption in a batch is quietly wrong. These are decoder-only models: with
    # right padding the pad tokens sit between the prompt and the first generated token, so
    # generation continues from padding instead of from the prompt. It raises nothing -- it returns
    # fluent, corrupted text with prompt fragments bleeding into captions, which is the worst
    # possible failure for a bulk pass that nobody reads in full.
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"loaded {args.model} as {type(model).__name__}", flush=True)

    done, t0 = 0, time.time()
    for i in range(0, len(todo), args.batch):
        chunk = todo[i: i + args.batch]
        images = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            im.thumbnail((args.side, args.side))
            images.append(im)
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": args.prompt}]}] for _ in chunk]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        trimmed = [o[len(i_):] for i_, o in zip(inputs.input_ids, out)]
        for p, cap in zip(chunk, proc.batch_decode(trimmed, skip_special_tokens=True)):
            if write_caption(p, cap):
                done += 1
        if done and (i // max(1, args.batch)) % 25 == 0:
            el = time.time() - t0
            rate = done / el
            print(f"{done:,}/{len(todo):,}  {rate:.2f} img/s  "
                  f"eta {(len(todo) - done) / max(rate, 1e-9) / 3600:.1f} h", flush=True)
    return done, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ext", default="webp")
    ap.add_argument("--fraction", type=int, default=10, help="1 in N images is recaptioned")
    ap.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    ap.add_argument("--model", required=True, help="ollama tag or HF id of the captioning VLM")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--prompt", default=PROMPT, help="override the captioning instruction")
    ap.add_argument("--workers", type=int, default=8, help="ollama: concurrent requests")
    ap.add_argument("--batch", type=int, default=8, help="transformers: batch size")
    ap.add_argument("--side", type=int, default=768)
    ap.add_argument("--max-new-tokens", type=int, default=110)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "**", f"*.{args.ext}"), recursive=True))
    picked = [p for p in files
              if selected(os.path.splitext(os.path.basename(p))[0], args.fraction)]
    todo = [p for p in picked if not os.path.exists(os.path.splitext(p)[0] + ".tags.txt")]
    print(f"corpus {len(files):,}  slice {len(picked):,} "
          f"({100 * len(picked) / max(1, len(files)):.1f}%)  remaining {len(todo):,}", flush=True)
    if args.dry_run:
        return
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do")
        return

    print(f"backend={args.backend} model={args.model}", flush=True)
    runner = run_ollama if args.backend == "ollama" else run_transformers
    done, el = runner(todo, args)
    print(f"done: {done:,} recaptioned in {el/3600:.2f} h ({done/max(el,1e-9):.2f} img/s)")


if __name__ == "__main__":
    main()
