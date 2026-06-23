#!/usr/bin/env python
"""LoRA trainer for Krea 2 (DiT adapters), cached latents. Train on Raw, run on Turbo.

    CUDA_VISIBLE_DEVICES=0 python train_t2i_lora_cached.py --config config/t2i_lora.yaml [--smoke]

The base DiT is frozen bf16; only the injected LoRA adapters train (standard AdamW, not the fused
full-FT path). Adapters save in ComfyUI/ai-toolkit key format (see lora.py) so they load on Turbo.
Reuses the shared data/preview helpers from the full-FT trainer.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time

import torch

from loading import build_dit, build_encoder, build_vae
from lora import inject_lora, inject_lora_te, load_lora_weights, lora_parameters, save_lora
from scheduler import get_schedule_for_seqlen
from train_t2i import edit_training_step, t2i_training_step
from train_t2i_full_joint_cached import (
    DEFAULT_PREVIEWS,
    home_volume_guard,
    index_caches,
    load_sample,
    preload_all,
    render_previews,
    resolve_resume_marker,
    rotate_by_glob,
    write_resume_marker,
)
from trackers import Tracker
from training_config import apply_runtime, dtype_of, load_config
from training_utils import (
    EmaModel,
    build_lr_scheduler,
    build_optimizer,
    compute_val_loss,
    derive_edit_mask,
    is_finite_loss,
    load_resume_state,
    save_resume_state,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)
    o, fl, lg, lc = cfg.optim, cfg.flow, cfg.logging, cfg.lora
    if lc.rank <= 0:
        raise SystemExit("lora.rank must be > 0 for the LoRA trainer (use train_t2i_full_joint_cached.py for FFT)")
    output_dir, ckpt_dir = cfg.paths.output_dir, cfg.paths.ckpt_dir
    home_volume_guard(ckpt_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_by_bucket, eval_paths = index_caches(cfg.paths.cache_dir, cfg.data.n_eval_holdout,
                                               train_list=cfg.data.train_list)
    bucket_keys = list(train_by_bucket)
    bucket_weights = [len(train_by_bucket[k]) for k in bucket_keys]
    if o.preload_caches:
        n_pre = preload_all([p for ps in train_by_bucket.values() for p in ps] + eval_paths)
        print(f"preloaded {n_pre} caches into RAM (optim.preload_caches)", flush=True)
    probe = next(iter(train_by_bucket.values()))[0]
    has_cached_text = "llm_text" in load_sample(probe, device)
    train_te = lc.te_rank > 0                    # TE-LoRA: adapt the Qwen3-VL text encoder (live encode)
    use_live_text = (not has_cached_text) or train_te
    print(f"LoRA rank={lc.rank} alpha={lc.alpha or lc.rank}: {sum(bucket_weights)} train / "
          f"{len(eval_paths)} eval, buckets={bucket_keys}, live_text={use_live_text}, "
          f"te_lora={train_te} train_dit={lc.train_transformer}", flush=True)

    dit = build_dit(cfg, device, dtype, load_weights=True, train=False)
    adapters = inject_lora(dit, lc.rank, lc.alpha, include_txtfusion=lc.target_txtfusion) \
        if lc.train_transformer else {}          # train_transformer=False -> TE-only (DiT frozen)
    dit.gradient_checkpointing = o.grad_checkpointing
    dit.train()  # enables grad-ckpt; base params stay frozen, only adapters require grad
    print(f"injected {len(adapters)} LoRA adapters", flush=True)
    if o.blocks_to_swap:
        # Page the deepest blocks' frozen base CPU<->GPU; LoRA adapters (trainable) stay GPU-resident.
        # Large VRAM saving for >1024 / larger LoRA. Needs grad-ckpt.
        swapped = dit.enable_block_swap(o.blocks_to_swap, device, skip_trainable=True)
        print(f"block-swap: paging {len(swapped)} deepest blocks CPU<->GPU "
              f"(frozen base only; adapters resident)", flush=True)

    encoder = build_encoder(cfg, device, dtype, train=False) if use_live_text else None
    te_adapters = {}
    if train_te:                                 # TE-LoRA: frozen Qwen3-VL base + trainable adapters
        te_adapters = inject_lora_te(encoder.qwen, lc.te_rank, lc.alpha)
        encoder.qwen.train()
        print(f"injected {len(te_adapters)} TE-LoRA adapters (Qwen3-VL)", flush=True)
    need_preview = bool(lg.sample_every) or args.smoke
    preview_encoder = encoder or (build_encoder(cfg, device, dtype, train=False) if need_preview else None)
    vae = build_vae(cfg, device, dtype) if need_preview else None

    params = lora_parameters(adapters) + lora_parameters(te_adapters)
    if not params:
        raise SystemExit("nothing to train: lora.train_transformer=False and lora.te_rank=0")
    opt = build_optimizer(o.optimizer, params, o.lr, weight_decay=o.weight_decay)
    sched_lr = build_lr_scheduler(opt, scheduler=o.lr_scheduler, warmup=o.warmup,
                                  total_steps=o.steps, min_lr_ratio=o.min_lr_ratio,
                                  num_restarts=o.num_restarts)

    gen = torch.Generator(device=device).manual_seed(cfg.runtime.seed)
    rng = random.Random(cfg.runtime.seed)
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    tracker = Tracker(lg.tracker, project=lg.wandb_project, run_name=lg.run_name or None, out_dir=output_dir)

    def collate(paths):
        samples = [load_sample(p, device) for p in paths]
        gh, gw = samples[0]["grid_h"], samples[0]["grid_w"]
        z0 = torch.stack([s["z_tgt"] for s in samples]).to(device, torch.float32)
        if use_live_text:
            ctx, mask = encoder([s["caption"] for s in samples])
            ctx, mask = ctx.to(device), mask.to(device)
        else:
            texts = [s["llm_text"] for s in samples]
            L = max(t.shape[0] for t in texts)
            ctx = torch.zeros(len(texts), L, texts[0].shape[1], texts[0].shape[2], dtype=texts[0].dtype)
            mask = torch.zeros(len(texts), L, dtype=torch.bool)
            for i, t in enumerate(texts):
                ctx[i, : t.shape[0]] = t
                mask[i, : t.shape[0]] = True
            ctx, mask = ctx.to(device), mask.to(device)
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

    # ----- EMA over the adapter tensors (cheap: only the LoRA params) -----
    ema = None
    if o.use_ema:
        named = {}
        for pool_tag, pool in (("dit", adapters), ("te", te_adapters)):
            for name, m in pool.items():
                named[f"{pool_tag}.{name}.lora_A"] = m.lora_A
                named[f"{pool_tag}.{name}.lora_B"] = m.lora_B
        ema = EmaModel(named, o.ema_decay, every=o.ema_every)
        print(f"EMA on: decay={o.ema_decay}, every={o.ema_every}", flush=True)

    # ----- Held-out validation: plain MSE at fixed schedule quantiles -----
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

    def _save_adapters(path, *, ema_tag=False):
        meta = {"variant": lc.variant}
        if ema_tag:
            meta["ema"] = "1"
        if adapters:
            save_lora(adapters, path, rank=lc.rank, alpha=lc.alpha or lc.rank, metadata=meta)
        if te_adapters:                          # TE-LoRA -> sibling .te.safetensors (text_encoder.* keys)
            save_lora(te_adapters, path.replace(".safetensors", ".te.safetensors"),
                      rank=lc.te_rank, alpha=lc.alpha or lc.te_rank, metadata=meta, key_prefix="text_encoder")

    # ----- Checkpoint = adapters + resume state (+ EMA adapters) + marker -----
    def checkpoint(step_n, *, tag=None):
        tag = tag or f"step{step_n:06d}"
        wpath = os.path.join(ckpt_dir, f"lora_{tag}.safetensors")
        _save_adapters(wpath)
        rotate_by_glob(os.path.join(ckpt_dir, "lora_step*.safetensors"), 2)
        spath = os.path.join(ckpt_dir, f"trainstate_{tag}.pt")
        save_resume_state(spath, step=step_n, optimizer=opt, scheduler=sched_lr, gen=gen, rng=rng, ema=ema)
        rotate_by_glob(os.path.join(ckpt_dir, "trainstate_step*.pt"), 2)
        epath = None
        if ema is not None:
            epath = os.path.join(ckpt_dir, f"lora_ema_{tag}.safetensors")
            ema.store()
            ema.copy_to()
            try:
                _save_adapters(epath, ema_tag=True)
            finally:
                ema.restore()
            rotate_by_glob(os.path.join(ckpt_dir, "lora_ema_step*.safetensors"), 2)
        write_resume_marker(ckpt_dir, step=step_n, weights=wpath, state=spath, ema=epath)
        return wpath

    # ----- Graceful interrupt: full resumable checkpoint, then exit -----
    _state = {"step": 0}

    def _on_signal(signum, frame):
        try:
            checkpoint(_state["step"], tag=f"interrupt{_state['step']:06d}")
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # ----- Resume: load adapters + optimizer + scheduler + RNG (+EMA) -----
    start_step = 0
    marker = resolve_resume_marker(cfg.paths.resume_from, ckpt_dir)
    if marker is not None:
        if adapters:
            load_lora_weights(adapters, marker["_weights_path"])
        if te_adapters:
            tep = marker["_weights_path"].replace(".safetensors", ".te.safetensors")
            if os.path.exists(tep):
                load_lora_weights(te_adapters, tep, key_prefix="text_encoder")
        start_step = load_resume_state(marker["_state_path"], optimizer=opt, scheduler=sched_lr,
                                       gen=gen, rng=rng, ema=ema)
        _state["step"] = start_step
        print(f"RESUMED at step {start_step} from {os.path.basename(marker['_weights_path'])}", flush=True)

    steps = 20 if args.smoke else o.steps
    print(f"training {steps} LoRA steps (optimizer={o.optimizer}, accum={o.accum})", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # peak_gb = training peak, not the one-time model-load spike
    t0 = time.time()
    run_loss = 0.0
    n_skipped = 0
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
        (loss / o.accum).backward()
        if (step + 1) % o.accum == 0:
            if o.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, o.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched_lr.step()
        run_loss += float(loss.detach())
        if ema is not None:
            ema.update(step)

        if (step + 1) % lg.log_every == 0:
            dt = (time.time() - t0) / lg.log_every
            rec = {"step": step + 1, "loss": run_loss / lg.log_every, "lr": sched_lr.get_last_lr()[0],
                   "s_per_step": round(dt, 3), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "skipped": n_skipped}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            tracker.log(rec, rec["step"])
            print(rec, flush=True)
            run_loss = 0.0
            t0 = time.time()
        if lg.val_every and (step + 1) % lg.val_every == 0 and val_groups:
            vrec = {"step": step + 1, "val_loss": run_val()}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(vrec) + "\n")
            tracker.log({"val_loss": vrec["val_loss"]}, vrec["step"])
            print({"step": vrec["step"], "val_loss": round(vrec["val_loss"], 5)}, flush=True)
            t0 = time.time()
        if lg.ckpt_every and (step + 1) % lg.ckpt_every == 0:
            checkpoint(step + 1)
        if lg.sample_every and (step + 1) % lg.sample_every == 0 and vae is not None and preview_encoder:
            render_previews(dit, vae, preview_encoder, DEFAULT_PREVIEWS[: lg.sample_count],
                            os.path.join(output_dir, "samples", f"step{step + 1:06d}_dashboard.png"),
                            res=cfg.data.resolution, steps=lg.sample_steps,
                            guidance=lg.sample_guidance, seed=cfg.runtime.seed)

    if args.smoke:
        checkpoint(steps, tag="smoke")
        if vae is not None and preview_encoder is not None:
            try:
                render_previews(dit, vae, preview_encoder, DEFAULT_PREVIEWS[:2],
                                os.path.join(output_dir, "samples", "lora_smoke_dashboard.png"),
                                res=cfg.data.resolution, steps=lg.sample_steps,
                                guidance=lg.sample_guidance, seed=cfg.runtime.seed)
            except Exception as e:
                print(f"[smoke] preview failed (non-fatal): {type(e).__name__} {e}", flush=True)
        print("SMOKE OK", flush=True)
    else:
        checkpoint(steps, tag="final")
    tracker.close()


if __name__ == "__main__":
    main()
