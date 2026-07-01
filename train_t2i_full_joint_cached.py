#!/usr/bin/env python
"""Full fine-tune trainer for Krea 2 (DiT, optionally + Qwen3-VL text encoder), cached latents.

Single-GPU:

    CUDA_VISIBLE_DEVICES=0 python train_t2i_full_joint_cached.py --config config/t2i_full.yaml [--smoke]

The memory recipe (fused per-parameter backward + on-GPU Adafactor + gradient checkpointing + bf16
stochastic rounding) fits the 12B DiT (+ optional 4B TE) on one 80GB GPU. See fused_adamw.py.

Two text paths, auto-selected:
  * cached  : caches hold ``llm_text`` and we are not training the TE -> DiT-only FFT, no encoder in loop.
  * live    : otherwise load the Qwen3-VL encoder (trainable iff te_lr>0) and re-encode captions each step.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import signal
import time

import torch

from fused_adamw import build_fused_adafactor, build_fused_adamw
from loading import build_dit, build_encoder, build_vae
from scheduler import get_schedule_for_seqlen
from train_t2i import edit_training_step, t2i_training_step
from trackers import Tracker
from training_config import apply_runtime, dtype_of, load_config
from training_utils import (
    EmaModel,
    build_lr_scheduler,
    compute_val_loss,
    derive_edit_mask,
    is_finite_loss,
    load_resume_state,
    save_resume_state,
)

# The fused per-parameter backward applies each update via a grad hook (not opt.step()), so torch's
# LR scheduler can't observe the optimizer step and emits a spurious one-time warning -- silence it.
import warnings as _warnings  # noqa: E402

_warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)`")

DEFAULT_PREVIEWS = [
    "a red fox walking through fresh snow, soft morning light",
    "a busy city street at night, neon signs, rain reflections",
    "a still life of fruit on a wooden table, studio lighting",
    "a lighthouse on a rocky coast under a stormy sky",
]


# --------------------------------------------------------------------------- #
# Cache dataset (bucketed by latent grid)
# --------------------------------------------------------------------------- #
def index_caches(cache_dir, n_eval, train_list=""):
    """Return (train_by_bucket: {(gh,gw): [paths]}, eval_paths). Eval holdout = idx < n_eval.

    ``train_list`` (optional): path to a JSON list of cache filenames; repeated names **oversample**.
    When given, the TRAIN set is built from that list (bucketed by latent grid); otherwise from every
    non-eval cache. The eval holdout is always ``idx < n_eval`` from the full cache dir.
    """
    files = sorted(glob.glob(os.path.join(cache_dir, "[0-9]" * 6 + ".pt")))
    if not files:
        raise SystemExit(f"no caches in {cache_dir}; run precache_t2i.py first")
    idx_path = os.path.join(cache_dir, ".bucket_index.json")
    bindex = {}
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            bindex = json.load(f)
    dirty = False
    for p in files:
        name = os.path.basename(p)
        st = os.stat(p)
        rec = bindex.get(name)
        if not rec or rec[2] != st.st_size:
            d = torch.load(p, map_location="cpu", weights_only=False)
            rec = [int(d["grid_h"]), int(d["grid_w"]), st.st_size]
            bindex[name] = rec
            dirty = True
    if dirty:
        tmp = idx_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(bindex, f)
        os.replace(tmp, idx_path)

    eval_paths = [p for p in files if int(os.path.basename(p)[:6]) < n_eval]
    if train_list:
        with open(train_list, "r", encoding="utf-8") as f:
            names = [os.path.basename(n) for n in json.load(f)]   # repeats -> oversampling
    else:
        names = [os.path.basename(p) for p in files if int(os.path.basename(p)[:6]) >= n_eval]
    train = {}
    for name in names:
        rec = bindex.get(name)
        if rec is None:
            print(f"[index_caches] {name} (train_list) not in {cache_dir}; skipping", flush=True)
            continue
        train.setdefault((rec[0], rec[1]), []).append(os.path.join(cache_dir, name))
    if not train:
        raise SystemExit("no training caches selected (check train_list / n_eval_holdout)")
    return train, eval_paths


# Optional RAM preload of cache payloads (optim.preload_caches): avoids per-step disk reads.
_PRELOAD: dict = {}


def load_sample(path, device):
    d = _PRELOAD.get(path)
    return d if d is not None else torch.load(path, map_location="cpu", weights_only=False)


def preload_all(paths) -> int:
    """Preload the given cache payloads into RAM (used when ``optim.preload_caches``)."""
    for p in set(paths):
        if p not in _PRELOAD:
            _PRELOAD[p] = torch.load(p, map_location="cpu", weights_only=False)
    return len(_PRELOAD)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def home_volume_guard(path):
    """Refuse to write checkpoints under / or /home or /root on Linux (use a data volume)."""
    if os.name != "posix":
        return
    ap = os.path.abspath(path)
    for bad in ("/home/", "/root/"):
        if ap.startswith(bad):
            raise SystemExit(f"refusing to write checkpoints under {bad} ({ap}); use a data volume")


def save_ckpt(model, ckpt_dir, tag, step, keep_last, *, prefix="dit"):
    from safetensors.torch import save_file

    os.makedirs(ckpt_dir, exist_ok=True)
    sd = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in model.state_dict().items()}
    out = os.path.join(ckpt_dir, f"{prefix}_{tag}.safetensors")
    tmp = out + ".tmp"
    save_file(sd, tmp, metadata={"step": str(step)})
    os.replace(tmp, out)
    # rotate step-tagged ckpts (named tags like 'final'/'interrupt' are kept)
    import re

    tagged = sorted(
        p for p in glob.glob(os.path.join(ckpt_dir, f"{prefix}_step*.safetensors"))
        if re.search(rf"{prefix}_step\d+\.safetensors$", p)
    )
    for old in tagged[:-keep_last]:
        os.remove(old)
    return out


def rotate_by_glob(pattern, keep_last):
    """Delete all but the newest ``keep_last`` step-tagged files matching ``pattern``.

    Only ``*step<N>.*`` files are eligible; named tags (``final`` / ``interrupt*``) are
    always kept, matching ``save_ckpt``'s own rotation policy.
    """
    import re

    files = sorted(p for p in glob.glob(pattern) if re.search(r"step\d+\.", os.path.basename(p)))
    for old in (files[:-keep_last] if keep_last > 0 else files):
        try:
            os.remove(old)
        except OSError:
            pass


def write_resume_marker(ckpt_dir, *, step, weights, state, ema=None, te=None):
    """Atomically (over)write resume.json -> the newest resumable checkpoint (basenames)."""
    marker = {
        "step": step,
        "weights": os.path.basename(weights),
        "state": os.path.basename(state),
        "ema": os.path.basename(ema) if ema else None,
        "te": os.path.basename(te) if te else None,
    }
    tmp = os.path.join(ckpt_dir, "resume.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(marker, f)
    os.replace(tmp, os.path.join(ckpt_dir, "resume.json"))


def resolve_resume_marker(resume_from, ckpt_dir):
    """Resolve ``paths.resume_from`` to a marker dict with absolute paths, or None.

    ``"auto"`` -> ``{ckpt_dir}/resume.json`` (None if absent = clean fresh start). An
    explicit value may name the resume.json or its directory; a missing explicit marker
    is a hard error (the user asked to resume a run that isn't there).
    """
    if not resume_from:
        return None
    path = os.path.join(ckpt_dir, "resume.json") if resume_from == "auto" else resume_from
    if os.path.isdir(path):
        path = os.path.join(path, "resume.json")
    if not os.path.exists(path):
        if resume_from == "auto":
            return None
        raise SystemExit(f"resume_from={resume_from!r}: no resume marker at {path}")
    with open(path, "r", encoding="utf-8") as f:
        marker = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    marker["_weights_path"] = os.path.join(base, marker["weights"])
    marker["_state_path"] = os.path.join(base, marker["state"])
    if marker.get("te"):
        marker["_te_path"] = os.path.join(base, marker["te"])
    return marker


# --------------------------------------------------------------------------- #
# Preview (reuse the reference sampler)
# --------------------------------------------------------------------------- #
def render_previews(dit, vae, encoder, prompts, out_path, *, res, steps, guidance, seed):
    from PIL import Image

    from sampling import sample

    dit.eval()
    try:
        imgs = sample(dit, vae, encoder, prompts, width=res, height=res,
                      steps=steps, guidance=guidance, seed=seed)
        cols = min(4, len(imgs))
        rows = (len(imgs) + cols - 1) // cols
        tw = imgs[0].width
        th = imgs[0].height
        sheet = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % cols) * tw, (i // cols) * th))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sheet.save(out_path)
    finally:
        dit.train()
        torch.cuda.empty_cache()


def render_edit_previews(dit, vae, encoder, examples, out_path, *, res, steps, guidance, seed):
    """Edit contact-sheet: one row per example, columns [source | model edit | target (if given)].

    ``examples`` = list of {"src": path, "instruction": str, "tgt": path?}. Rendered at the training
    resolution (square, matching precache_edit) so a step-0 render is the base model's edit attempt
    and later sheets show the same fixed edits improving toward the ground-truth target column.
    """
    from PIL import Image

    from sample_edit import edit_sample

    dit.eval()
    try:
        rows = []
        for ex in examples:
            src = Image.open(ex["src"]).convert("RGB")
            edited = edit_sample(dit, vae, encoder, ex["instruction"], [src],
                                 width=res, height=res, steps=steps, guidance=guidance, seed=seed)
            cells = [src.resize((res, res)), edited]
            if ex.get("tgt") and os.path.exists(ex["tgt"]):
                cells.append(Image.open(ex["tgt"]).convert("RGB").resize((res, res)))
            rows.append(cells)
        ncol = max(len(r) for r in rows)
        sheet = Image.new("RGB", (ncol * res, len(rows) * res), (0, 0, 0))
        for ri, r in enumerate(rows):
            for ci, im in enumerate(r):
                sheet.paste(im, (ci * res, ri * res))
        sdir = os.path.dirname(out_path)
        os.makedirs(sdir, exist_ok=True)
        sheet.save(out_path)
        # Sidecars so the dashboard splits this sheet into one captioned card per example
        # (rows = examples; layout 'edit' makes /tile slice by row). Cheap; rewritten each call.
        with open(os.path.join(sdir, "prompts.json"), "w", encoding="utf-8") as pf:
            json.dump({i: ex.get("instruction", "") for i, ex in enumerate(examples)}, pf,
                      ensure_ascii=False)
        with open(os.path.join(sdir, "layout.json"), "w", encoding="utf-8") as lf:
            json.dump({"mode": "edit", "cols": ncol}, lf)
    finally:
        dit.train()
        torch.cuda.empty_cache()


def slice_base_tiles(sheet_path, base_root, n):
    """Slice a base edit contact-sheet (``n`` example rows) into per-example BASE tiles
    ``idx0..n-1`` that the dashboard serves for its base-vs-current toggle. Stale tiles are
    cleared first so a shrunk/reordered manifest never leaves orphans. No-op if the sheet
    is missing."""
    from PIL import Image

    if not os.path.exists(sheet_path):
        return
    os.makedirs(base_root, exist_ok=True)
    for old in glob.glob(os.path.join(base_root, "idx*.png")):
        os.remove(old)
    sheet = Image.open(sheet_path)
    W, H = sheet.size
    th = H // max(1, n)
    for k in range(n):
        sheet.crop((0, k * th, W, k * th + th)).save(os.path.join(base_root, f"idx{k}.png"))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="run ~20 steps + one preview, then exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)
    o, fl, lg = cfg.optim, cfg.flow, cfg.logging
    output_dir, ckpt_dir = cfg.paths.output_dir, cfg.paths.ckpt_dir
    home_volume_guard(ckpt_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_by_bucket, eval_paths = index_caches(cfg.paths.cache_dir, cfg.data.n_eval_holdout,
                                               train_list=cfg.data.train_list)
    bucket_keys = list(train_by_bucket)
    bucket_weights = [len(train_by_bucket[k]) for k in bucket_keys]
    n_train = sum(bucket_weights)
    if o.preload_caches:
        n_pre = preload_all([p for ps in train_by_bucket.values() for p in ps] + eval_paths)
        print(f"preloaded {n_pre} caches into RAM (optim.preload_caches)", flush=True)
    probe = next(iter(train_by_bucket.values()))[0]
    has_cached_text = "llm_text" in load_sample(probe, device)

    # Decide text path. Two INDEPENDENT questions (don't conflate them):
    #   train the TE?  -> only when te_lr>0 (explicit joint FFT); TE-FFT on narrow data risks forgetting.
    #   live-encode?   -> whenever the TE trains OR the cache holds no text. The latter is the
    #                     latents-only cache: a FROZEN Qwen3-VL re-encodes the stored caption each step
    #                     (same signal as caching, but no ~3.8MB/prompt 12-layer text on disk).
    train_te = (o.te_lr > 0.0)
    use_live_text = train_te or (not has_cached_text)
    te_lr = o.te_lr if o.te_lr > 0 else 0.0
    print(f"caches: {n_train} train / {len(eval_paths)} eval, buckets={bucket_keys}, "
          f"cached_text={has_cached_text}, train_te={train_te}, live_text={use_live_text}", flush=True)

    dit = build_dit(cfg, device, dtype, load_weights=True, train=o.train_dit)
    dit.gradient_checkpointing = o.grad_checkpointing
    if not o.train_dit:
        dit.eval().requires_grad_(False)
    if o.blocks_to_swap:
        # Page the deepest blocks CPU<->GPU. FFT (train_dit) must page trainable weights too (slow);
        # DiT-frozen runs keep them resident. Block swap is most useful in the LoRA trainer (frozen base).
        swapped = dit.enable_block_swap(o.blocks_to_swap, device, skip_trainable=not o.train_dit)
        print(f"block-swap: paging {len(swapped)} deepest blocks CPU<->GPU "
              f"(skip_trainable={not o.train_dit})", flush=True)
    encoder = None
    preview_encoder = None
    if use_live_text:
        encoder = build_encoder(cfg, device, dtype, train=train_te)
        preview_encoder = encoder
    need_preview = bool(lg.sample_every) or args.smoke
    vae = build_vae(cfg, device, dtype) if need_preview else None
    if need_preview and preview_encoder is None:
        # DiT-only training (cached text): still need a frozen encoder to render previews.
        preview_encoder = build_encoder(cfg, device, dtype, train=False)

    # Preview set: curated edit examples ([src|edit|tgt] rows) when a manifest is given, else t2i prompts.
    edit_examples = None
    if need_preview and lg.edit_preview_manifest:
        with open(lg.edit_preview_manifest, encoding="utf-8") as _f:
            edit_examples = [json.loads(line) for line in _f if line.strip()]
        print(f"edit previews: {len(edit_examples)} curated examples from {lg.edit_preview_manifest}", flush=True)

    def do_preview(tag):
        if vae is None or preview_encoder is None:
            return
        out = os.path.join(output_dir, "samples", f"{tag}.png")
        try:
            if edit_examples:
                render_edit_previews(dit, vae, preview_encoder, edit_examples, out,
                                     res=cfg.data.resolution, steps=lg.sample_steps,
                                     guidance=lg.sample_guidance, seed=cfg.runtime.seed)
            else:
                render_previews(dit, vae, preview_encoder, DEFAULT_PREVIEWS[: lg.sample_count], out,
                                res=cfg.data.resolution, steps=lg.sample_steps,
                                guidance=lg.sample_guidance, seed=cfg.runtime.seed)
        except Exception as e:
            print(f"[preview] {tag} failed (non-fatal): {type(e).__name__} {e}", flush=True)

    # Param groups (separate LR for DiT vs TE).
    groups = []
    if o.train_dit:
        groups.append({"params": [p for p in dit.parameters() if p.requires_grad], "lr": o.lr, "name": "dit"})
    if train_te and encoder is not None:
        groups.append({"params": [p for p in encoder.qwen.parameters() if p.requires_grad], "lr": te_lr, "name": "te"})
    if not groups:
        raise SystemExit("nothing to train (train_dit=False and train_te=False)")

    if o.optimizer_state == "adafactor":
        opt = build_fused_adafactor(groups, lr=o.lr)
    else:
        opt = build_fused_adamw(groups, lr=o.lr, weight_decay=o.weight_decay,
                                offload_states=o.offload_optimizer)
    sched_lr = build_lr_scheduler(opt, scheduler=o.lr_scheduler, warmup=o.warmup,
                                  total_steps=o.steps, min_lr_ratio=o.min_lr_ratio,
                                  num_restarts=o.num_restarts)

    # Positional group indices (order is fixed: dit first if trained, then te). Used for the
    # live-group lookup below and for reading te's LR at log time -- both robust to
    # load_state_dict, which replaces the param-group dicts on resume (see the hook note).
    te_gidx = (len(groups) - 1) if (train_te and encoder is not None) else None

    # Per-step, per-group grad-norm accumulator (sum of per-param grad-norm^2 within a step).
    # Folded into the interval mean at each log flush -> per-group learning-signal metric
    # (e.g. te_grad_norm shows the jointly-trained TE is actually receiving/using gradient).
    gstep = {"dit": 0.0, "te": 0.0}

    # Fused per-parameter backward (accum==1): step + free each grad as it lands.
    fused = (o.accum == 1)
    if fused:
        for gidx, group in enumerate(opt.param_groups):
            nm = group.get("name", "dit")           # captured as a str -> survives load_state_dict
            for p in group["params"]:
                def _hook(param, gidx=gidx, nm=nm):
                    # Look up the LIVE group by index: load_state_dict (resume) swaps the group
                    # dicts, so a captured dict would carry a stale (frozen) LR. Indexing
                    # opt.param_groups keeps step_parameter on the scheduler-updated group.
                    g = opt.param_groups[gidx]
                    if param.grad is None or not torch.isfinite(param.grad).all():
                        param.grad = None
                        return
                    # clip_grad_norm_ already returns the pre-clip norm -> capture it for free.
                    if o.grad_clip:
                        gn = torch.nn.utils.clip_grad_norm_(param, o.grad_clip)
                    else:
                        gn = param.grad.detach().norm()
                    gstep[nm] += float(gn) ** 2
                    opt.step_parameter(param, g, 0)
                    param.grad = None
                p.register_post_accumulate_grad_hook(_hook)

    gen = torch.Generator(device=device).manual_seed(cfg.runtime.seed)
    rng = random.Random(cfg.runtime.seed)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    tracker = Tracker(lg.tracker, project=lg.wandb_project, run_name=lg.run_name or None,
                      out_dir=output_dir)

    def collate(paths):
        """Build (z0, context, text_mask, grid_h, grid_w, refs, ref_grids) for a grid-homogeneous batch."""
        samples = [load_sample(p, device) for p in paths]
        gh, gw = samples[0]["grid_h"], samples[0]["grid_w"]
        z0 = torch.stack([s["z_tgt"] for s in samples]).to(device, torch.float32)  # (B,n,64)
        if use_live_text:
            caps = [s["caption"] for s in samples]
            ctx, mask = encoder(caps)                       # (B,L,12,2560), (B,L)
            ctx = ctx.to(device)
            mask = mask.to(device)
        else:
            texts = [s["llm_text"] for s in samples]        # each (n_i,12,2560)
            L = max(t.shape[0] for t in texts)
            B = len(texts)
            ctx = torch.zeros(B, L, texts[0].shape[1], texts[0].shape[2], dtype=texts[0].dtype)
            mask = torch.zeros(B, L, dtype=torch.bool)
            for i, t in enumerate(texts):
                ctx[i, : t.shape[0]] = t
                mask[i, : t.shape[0]] = True
            ctx = ctx.to(device)
            mask = mask.to(device)
        refs = ref_grids = None
        if samples[0].get("refs"):                          # edit / multiref / style cache
            ref_grids = [(r["grid_h"], r["grid_w"]) for r in samples[0]["refs"]]
            refs = [torch.stack([s["refs"][i]["tokens"] for s in samples]).to(device, torch.float32)
                    for i in range(len(ref_grids))]
        return z0, ctx, mask, gh, gw, refs, ref_grids

    def sample_batch():
        key = rng.choices(bucket_keys, weights=bucket_weights, k=1)[0]
        files = train_by_bucket[key]
        return collate([rng.choice(files) for _ in range(o.batch)])

    # ----- EMA (CPU fp32 shadow of the trained DiT weights; zero extra VRAM) -----
    ema = None
    if o.use_ema and o.train_dit:
        ema = EmaModel({k: v for k, v in dit.state_dict().items() if v.is_floating_point()},
                       o.ema_decay, every=o.ema_every)
        print(f"EMA on: decay={o.ema_decay}, every={o.ema_every} (CPU fp32 shadow)", flush=True)

    # ----- Held-out validation: plain MSE at fixed schedule quantiles (low variance) -----
    val_groups = {}
    for p in eval_paths:
        d = load_sample(p, device)
        val_groups.setdefault((int(d["grid_h"]), int(d["grid_w"])), []).append(p)

    def run_val():
        was_training = dit.training
        dit.eval()
        with torch.no_grad():
            batches = []
            for (gh, gw), paths in val_groups.items():
                vsched = get_schedule_for_seqlen(gh * gw, sigma=fl.sigma, base_shift=fl.base_shift,
                                                 max_shift=fl.max_shift, base_seq_len=fl.base_image_seq_len,
                                                 max_seq_len=fl.max_image_seq_len)
                for i in range(0, len(paths), o.batch):
                    chunk = paths[i:i + o.batch]
                    batches.append(((collate(chunk), vsched), len(chunk)))

            def loss_fn(packed, q):
                (z0, ctx, mask, gh, gw, refs, ref_grids), vsched = packed
                t = float(vsched(torch.tensor([float(q)])).item())
                g = torch.Generator(device=device).manual_seed(cfg.runtime.seed + int(round(q * 1000)))
                if refs is not None:
                    return edit_training_step(dit, z0=z0, refs=refs, ref_grids=ref_grids, context=ctx,
                                              text_mask=mask, grid_h=gh, grid_w=gw, schedule=vsched,
                                              flow_cfg=fl, generator=g, cfg_dropout_prob=0.0,
                                              ref_dropout_prob=0.0, t_override=t, disable_weighting=True)
                return t2i_training_step(dit, z0=z0, context=ctx, text_mask=mask, grid_h=gh, grid_w=gw,
                                         schedule=vsched, flow_cfg=fl, generator=g, cfg_dropout_prob=0.0,
                                         t_override=t, disable_weighting=True)

            v = compute_val_loss(loss_fn, batches)
        if was_training:
            dit.train()
        return v

    # ----- Checkpoint = DiT weights (+TE) + resume state (+EMA weights) + marker -----
    def checkpoint(step_n, *, tag=None):
        tag = tag or f"step{step_n:06d}"
        wpath = save_ckpt(dit, ckpt_dir, tag, step_n, keep_last=2, prefix="dit")
        te_path = None
        if train_te and encoder is not None:
            te_path = save_ckpt(encoder.qwen, ckpt_dir, tag, step_n, keep_last=2, prefix="te")
        spath = os.path.join(ckpt_dir, f"trainstate_{tag}.pt")
        save_resume_state(spath, step=step_n, optimizer=opt, scheduler=sched_lr, gen=gen, rng=rng, ema=ema)
        rotate_by_glob(os.path.join(ckpt_dir, "trainstate_step*.pt"), 2)
        epath = None
        if ema is not None:
            epath = os.path.join(ckpt_dir, f"dit_ema_{tag}.safetensors")
            ema.write_safetensors(epath, dit.state_dict(), metadata={"step": str(step_n), "ema": "1"})
            rotate_by_glob(os.path.join(ckpt_dir, "dit_ema_step*.safetensors"), 2)
        write_resume_marker(ckpt_dir, step=step_n, weights=wpath, state=spath, ema=epath, te=te_path)
        return wpath

    # ----- Graceful interrupt: save a FULL resumable checkpoint, then exit (survives SSH drop) -----
    _state = {"step": 0}

    def _on_signal(signum, frame):
        try:
            if o.train_dit:
                checkpoint(_state["step"], tag=f"interrupt{_state['step']:06d}")
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # ----- Base edit-preview baseline (BEFORE any resume weight-load, while the DiT still holds
    # base weights): render the untrained model's edit sheet + slice per-example BASE tiles so the
    # dashboard's base-vs-current toggle stays aligned with the CURRENT manifest. Regenerated only
    # when the tiles are missing or their count no longer matches the manifest (e.g. the preview set
    # was edited), so a resume with an unchanged manifest skips it and a resume after editing the
    # set refreshes the base counterparts instead of leaving them stale. -----
    if lg.sample_every and edit_examples:
        base_root = os.path.join(output_dir, "base_previews")
        if len(glob.glob(os.path.join(base_root, "idx*.png"))) != len(edit_examples):
            base_sheet = os.path.join(output_dir, "samples", "step000000_base.png")
            do_preview("step000000_base")
            slice_base_tiles(base_sheet, base_root, len(edit_examples))

    # ----- Resume: load weights (+TE) + optimizer + scheduler + RNG (+EMA), continue from step -----
    start_step = 0
    marker = resolve_resume_marker(cfg.paths.resume_from, ckpt_dir)
    if marker is not None:
        from safetensors.torch import load_file

        dit.load_state_dict(load_file(marker["_weights_path"]), strict=True)
        if marker.get("_te_path") and encoder is not None:
            encoder.qwen.load_state_dict(load_file(marker["_te_path"]), strict=False)
        start_step = load_resume_state(marker["_state_path"], optimizer=opt, scheduler=sched_lr,
                                       gen=gen, rng=rng, ema=ema, offload=o.offload_optimizer)
        _state["step"] = start_step
        print(f"RESUMED at step {start_step} from {os.path.basename(marker['_weights_path'])}", flush=True)

    steps = 20 if args.smoke else o.steps
    print(f"training {steps} steps (fused={fused}, optim_state={o.optimizer_state}, te_lr={te_lr})", flush=True)
    # Step-0 baseline for t2i previews (edit-mode base is handled before the resume load above).
    if lg.sample_every and start_step == 0 and not edit_examples:
        do_preview("step000000_base")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # peak_gb = training peak, not the one-time model-load spike
    t0 = time.time()
    run_loss = 0.0
    n_skipped = 0
    gsum = {"dit": 0.0, "te": 0.0}   # interval sum of per-step total grad norms (per group)
    gcnt = 0                          # number of optimizer steps folded into gsum this interval

    for step in range(start_step, steps):
        _state["step"] = step
        z0, ctx, mask, gh, gw, refs, ref_grids = sample_batch()
        schedule = get_schedule_for_seqlen(gh * gw, sigma=fl.sigma, base_shift=fl.base_shift,
                                           max_shift=fl.max_shift, base_seq_len=fl.base_image_seq_len,
                                           max_seq_len=fl.max_image_seq_len)
        if refs is not None:
            lmask = None
            if cfg.data.masked_loss and refs[0].shape[1] == z0.shape[1]:
                lmask = derive_edit_mask(refs[0], z0, quantile=cfg.data.mask_quantile)
                if cfg.data.mask_bg_weight:
                    lmask = lmask + (1.0 - lmask) * cfg.data.mask_bg_weight
            loss = edit_training_step(dit, z0=z0, refs=refs, ref_grids=ref_grids, context=ctx,
                                      text_mask=mask, grid_h=gh, grid_w=gw, schedule=schedule,
                                      flow_cfg=fl, generator=gen, cfg_dropout_prob=o.cfg_dropout_prob,
                                      ref_dropout_prob=cfg.data.ref_dropout_prob, loss_mask=lmask)
        else:
            loss = t2i_training_step(dit, z0=z0, context=ctx, text_mask=mask, grid_h=gh, grid_w=gw,
                                     schedule=schedule, flow_cfg=fl, generator=gen,
                                     cfg_dropout_prob=o.cfg_dropout_prob)
        if not is_finite_loss(loss):
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            sched_lr.step()
            continue

        if fused:
            loss.backward()                 # hooks step + free each grad (also fills gstep)
        else:
            (loss / o.accum).backward()
            if (step + 1) % o.accum == 0:
                for group in opt.param_groups:
                    for p in group["params"]:
                        if p.grad is not None and torch.isfinite(p.grad).all():
                            if o.grad_clip:
                                gn = torch.nn.utils.clip_grad_norm_(p, o.grad_clip)
                            else:
                                gn = p.grad.detach().norm()
                            gstep[group.get("name", "dit")] += float(gn) ** 2
                            opt.step_parameter(p, group, 0)
                        if p.grad is not None:
                            p.grad = None
        # Fold this step's per-group total grad norm (sqrt of summed per-param norm^2) into the
        # interval mean, then reset the per-step accumulator. Only when an optimizer step happened.
        if fused or (step + 1) % o.accum == 0:
            for _nm in gstep:
                gsum[_nm] += gstep[_nm] ** 0.5
                gstep[_nm] = 0.0
            gcnt += 1
        sched_lr.step()
        run_loss += float(loss.detach())
        if ema is not None:
            ema.update(step)

        if (step + 1) % lg.log_every == 0:
            dt = (time.time() - t0) / lg.log_every
            rec = {"step": step + 1, "loss": run_loss / lg.log_every,
                   "lr": sched_lr.get_last_lr()[0], "s_per_step": round(dt, 3),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "skipped": n_skipped}
            # Per-group grad-norm (mean over the interval). There is ONE joint flow loss (no
            # separate TE loss exists); te_grad_norm is the honest "is the TE learning" signal.
            if gcnt:
                rec["dit_grad_norm"] = round(gsum["dit"] / gcnt, 4)
                if train_te and te_gidx is not None:
                    # Read the live per-group LR from the optimizer (get_last_lr()'s cached
                    # _last_lr is not restored on resume; opt.param_groups is the source of truth).
                    rec["te_grad_norm"] = round(gsum["te"] / gcnt, 4)
                    rec["te_lr"] = opt.param_groups[te_gidx]["lr"]
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            tracker.log(rec, rec["step"])
            print(rec, flush=True)
            run_loss = 0.0
            gsum = {"dit": 0.0, "te": 0.0}
            gcnt = 0
            t0 = time.time()

        if lg.val_every and (step + 1) % lg.val_every == 0 and val_groups:
            vrec = {"step": step + 1, "val_loss": run_val()}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(vrec) + "\n")
            tracker.log({"val_loss": vrec["val_loss"]}, vrec["step"])
            print({"step": vrec["step"], "val_loss": round(vrec["val_loss"], 5)}, flush=True)
            t0 = time.time()  # don't bill val wall-time to the next step's s/step

        if lg.ckpt_every and (step + 1) % lg.ckpt_every == 0 and o.train_dit:
            checkpoint(step + 1)

        if lg.sample_every and (step + 1) % lg.sample_every == 0:
            do_preview(f"step{step + 1:06d}")

    if args.smoke:
        do_preview("smoke")
        print("SMOKE OK", flush=True)
    elif o.train_dit:
        checkpoint(steps, tag="final")
    tracker.close()


if __name__ == "__main__":
    main()
