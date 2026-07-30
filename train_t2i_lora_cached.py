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
from contextlib import nullcontext

import torch

from loading import build_dit, build_encoder, build_vae
from lora import (ema_named_tensors, inject_lora, inject_lora_te, load_lora_stage_init,
                  load_lora_weights, lora_disabled, lora_parameters, save_lora)
from scheduler import get_schedule_for_seqlen
from train_t2i import (augment_reference_order, edit_training_step,
                        encode_text_with_ref_dropout, strip_trigger, t2i_training_step)
from train_t2i_full_joint_cached import (
    DEFAULT_PREVIEWS,
    home_volume_guard,
    index_caches,
    load_cache_annotations,
    load_sample,
    preload_all,
    reference_delta_flags,
    render_edit_previews,
    render_previews,
    cache_edit_ref_geometry,
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
    is_finite_loss,
    load_resume_state,
    reference_delta_loss_mask,
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
    marker = resolve_resume_marker(cfg.paths.resume_from, ckpt_dir)

    cache_annotations, _ = load_cache_annotations(cfg.data.cache_annotations)
    train_by_bucket, eval_paths = index_caches(
        cfg.paths.cache_dir, cfg.data.n_eval_holdout,
        train_list=cfg.data.train_list, eval_list=cfg.data.eval_list)
    bucket_keys = list(train_by_bucket)
    bucket_weights = [len(train_by_bucket[k]) for k in bucket_keys]
    if o.preload_caches:
        n_pre = preload_all([p for ps in train_by_bucket.values() for p in ps] + eval_paths)
        print(f"preloaded {n_pre} caches into RAM (optim.preload_caches)", flush=True)
    probe = next(iter(train_by_bucket.values()))[0]
    has_cached_text = "llm_text" in load_sample(probe, device)
    train_te = lc.te_rank > 0                    # TE-LoRA: adapt the Qwen3-VL text encoder (live encode)
    use_live_text = (not has_cached_text) or train_te
    if cfg.data.edit_vlm_cond and has_cached_text:
        raise SystemExit(
            "data.edit_vlm_cond requires a latents-only edit cache; rebuild with "
            "precache_edit.py --no-cache-text so grounded text is encoded live")
    print(f"LoRA variant={lc.variant} rank={lc.rank} alpha={lc.alpha or lc.rank}: "
          f"{sum(bucket_weights)} train / "
          f"{len(eval_paths)} eval, buckets={bucket_keys}, live_text={use_live_text}, "
          f"te_lora={train_te} train_dit={lc.train_transformer}", flush=True)

    dit = build_dit(cfg, device, dtype, load_weights=True, train=False)
    quant = o.quantize_base if lc.train_transformer else ""
    fp8_base = quant == "fp8"
    if fp8_base:
        from quantize import quantize_dit_fp8   # quantise the frozen base to e4m3 to lower its memory
        nq = quantize_dit_fp8(dit)
        print(f"fp8-quantized {nq} frozen base Linears (attn+mlp)", flush=True)
    elif quant == "int8":
        from convrot_int8 import int8_gemm_supported, quantize_dit_int8

        if not int8_gemm_supported(device):
            raise SystemExit("quantize_base=int8 needs int8 tensor cores (Ampere or newer)")
        nq = quantize_dit_int8(dit)
        print(f"int8-quantized {nq} frozen base Linears (attn+mlp)", flush=True)
    elif quant:
        raise SystemExit(f"unknown optim.quantize_base {quant!r}; expected '', 'fp8' or 'int8'")
    adapters = inject_lora(dit, lc.rank, lc.alpha,
                           include_txtfusion=lc.target_txtfusion,
                           include_txtmlp=lc.target_txtmlp,
                           variant=lc.variant) \
        if lc.train_transformer else {}          # train_transformer=False -> TE-only (DiT frozen)
    # fp8 requires gradient checkpointing: otherwise the per-forward dequant is retained for backward
    # across every layer, erasing the memory saving. Force it on when fp8 is active. int8 does not
    # need it -- its backward re-derives the weight from the stored codes and never holds a
    # dequantized copy.
    if fp8_base and not o.grad_checkpointing:
        print("note: enabling gradient checkpointing (required by the fp8 base)", flush=True)
    dit.gradient_checkpointing = o.grad_checkpointing or fp8_base
    dit.train()  # enables grad-ckpt; base params stay frozen, only adapters require grad
    print(f"injected {len(adapters)} LoRA adapters", flush=True)
    if o.compile_blocks:
        if o.blocks_to_swap:
            raise SystemExit("optim.compile_blocks and optim.blocks_to_swap cannot be combined")
        if not lc.train_transformer:
            raise SystemExit("optim.compile_blocks requires lora.train_transformer=true")
        ncompiled = dit.compile_blocks(o.compile_mode)
        print(f"compiled {ncompiled} transformer blocks (mode={o.compile_mode})", flush=True)
    if o.blocks_to_swap:
        # Page the deepest blocks' frozen base CPU<->GPU; LoRA adapters (trainable) stay GPU-resident.
        # Large VRAM saving for >1024 / larger LoRA. Needs grad-ckpt.
        swapped = dit.enable_block_swap(o.blocks_to_swap, device, skip_trainable=True)
        print(f"block-swap: paging {len(swapped)} deepest blocks CPU<->GPU "
              f"(frozen base only; adapters resident)", flush=True)

    encoder = build_encoder(cfg, device, dtype, train=False) if use_live_text else None
    te_adapters = {}
    quant_te = str(o.quantize_te or "").lower()
    if quant_te not in ("", "fp8"):
        raise SystemExit(f"unknown optim.quantize_te {o.quantize_te!r}; expected '' or 'fp8'")
    if quant_te and train_te:
        raise SystemExit("optim.quantize_te is only supported for a frozen text encoder, not TE-LoRA")
    if encoder is not None and quant_te == "fp8":
        from quantize import quantize_frozen_linears_fp8
        nq = quantize_frozen_linears_fp8(encoder.qwen)
        print(f"fp8-quantized {nq} frozen Qwen3-VL Linears", flush=True)
    if train_te:                                 # TE-LoRA: frozen Qwen3-VL base + trainable adapters
        te_adapters = inject_lora_te(encoder.qwen, lc.te_rank, lc.alpha, variant=lc.variant)
        encoder.qwen.train()
        print(f"injected {len(te_adapters)} TE-LoRA adapters (Qwen3-VL, variant={lc.variant})",
              flush=True)
    if marker is None and cfg.paths.lora_init:
        load_lora_stage_init(adapters, te_adapters, cfg.paths.lora_init)
        print(
            f"LoRA stage warm start: {os.path.basename(cfg.paths.lora_init)} "
            "(fresh optimizer, scheduler, RNG and step 0)",
            flush=True,
        )
    need_preview = bool(lg.sample_every) or args.smoke
    preview_encoder = encoder or (build_encoder(cfg, device, dtype, train=False) if need_preview else None)
    vae = build_vae(cfg, device, dtype) if need_preview else None

    # ----- Differential output preservation: prior-prompt conditioning -----
    dp = cfg.dop
    dop_on = bool(dp.enabled) and float(dp.weight) != 0.0
    dop_encoder, dop_fixed = None, None
    if dop_on:
        if dp.every < 1:
            raise SystemExit("dop.every must be >= 1")
        if not adapters:
            raise SystemExit(
                "dop.enabled requires lora.train_transformer=true: the preservation target is the "
                "DiT's own prediction with its adapters disabled, so with a frozen DiT the term is "
                "identically zero")
        if bool(dp.prompt) == bool(dp.trigger):
            raise SystemExit("set exactly one of dop.prompt (fixed prior prompt) or dop.trigger "
                             "(strip the trigger word from each caption)")
        # A prior prompt is text we choose at train time, so it must be ENCODED -- a latents-only
        # cache cannot contain it. Reuse whichever encoder is already loaded rather than a third copy.
        dop_encoder = encoder or preview_encoder
        if dop_encoder is None:
            dop_encoder = build_encoder(cfg, device, dtype, train=False)
            print("dop: loaded a frozen text encoder to encode the prior prompt", flush=True)
        if dp.trigger and "caption" not in load_sample(probe, device):
            raise SystemExit(
                "dop.trigger needs the caption in the cache to strip the trigger from; this cache has "
                "none. Use dop.prompt with a fixed class prompt, or rebuild the cache with captions")
        if dp.prompt:
            # Fixed prompt -> encode once. Also encoded with the TE adapters OFF when TE-LoRA is on,
            # so the prior conditioning is a constant of the BASE model and cannot drift with training
            # (the term then constrains the DiT, which is what owns the prior being preserved).
            with torch.no_grad(), (lora_disabled(dop_encoder.qwen) if te_adapters else nullcontext()):
                pc, pm = dop_encoder([dp.prompt])
            dop_fixed = (pc.to(device).detach(), pm.to(device).detach())
        print(f"dop: ON weight={dp.weight} every={dp.every} "
              f"{'prompt=' + repr(dp.prompt) if dp.prompt else 'trigger=' + repr(dp.trigger)} "
              f"(~2x step cost)", flush=True)

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
                                     guidance=lg.sample_guidance, seed=cfg.runtime.seed,
                                     vlm_cond=cfg.data.edit_vlm_cond,
                                     vlm_image_size=cfg.data.vlm_image_size,
                                     edit_ref_geometry=cfg.data.edit_ref_geometry,
                                     fit_ref_crop_tolerance=cfg.data.fit_ref_crop_tolerance)
            else:
                render_previews(dit, vae, preview_encoder, DEFAULT_PREVIEWS[: lg.sample_count], out,
                                res=cfg.data.resolution, steps=lg.sample_steps,
                                guidance=lg.sample_guidance, seed=cfg.runtime.seed)
        except Exception as e:
            print(f"[preview] {tag} failed (non-fatal): {type(e).__name__} {e}", flush=True)

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

    def collate(paths, drop_refs=False, augment_order=False, dop=False):
        # drop_refs (train batches only): with VLM dual conditioning, reference dropout is decided here
        # so the text encoding agrees with the latent refs — dropped samples are encoded text-only and
        # the mask travels to edit_training_step as ref_drop. Validation keeps it off.
        samples = [load_sample(p, device) for p in paths]
        delta_mask_flags = reference_delta_flags(samples, paths, cache_annotations)
        samples = augment_reference_order(
            samples, rng, cfg.data.ref_permute_prob, enabled=augment_order)
        gh, gw = samples[0]["grid_h"], samples[0]["grid_w"]
        z0 = torch.stack([s["z_tgt"] for s in samples]).to(device, torch.float32)
        ref_drop = None
        # Hoisted: the prior-prompt encoding below reuses the same captions and the same VLM image
        # groups, so the prior differs from the training conditioning ONLY in the text.
        caps = [s.get("caption", "") for s in samples]
        vlm_imgs = None
        if use_live_text:
            if cfg.data.edit_vlm_cond and samples[0].get("ref_paths"):
                from PIL import Image

                from sample_edit import _vlm_resize
                # One GROUP of images per sample, in the same order the refs are packed as latents,
                # so a caption saying "image_2" resolves to refs[1] in both the text encoder and the
                # DiT stream. Passing only ref_paths[0] leaves multi-reference composition
                # instructions ungroundable (the encoder never sees the second subject).
                vlm_imgs = [[_vlm_resize(
                                 Image.open(p_), cfg.data.vlm_image_size,
                                 cfg.data.vlm_image_min_size, rng)
                             for p_ in s["ref_paths"]] for s in samples]
            if vlm_imgs is not None and drop_refs and cfg.data.ref_dropout_prob > 0.0:
                ref_drop = torch.rand(len(samples)) < cfg.data.ref_dropout_prob
                ctx, mask = encode_text_with_ref_dropout(encoder, caps, vlm_imgs, ref_drop)
            else:
                ctx, mask = encoder(caps, images=vlm_imgs)  # images=None -> text-only (unchanged)
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
            # Caches are bucketed by TARGET grid only, so a batch can mix reference COUNTS; taking
            # the signature from sample 0 would raise a bare IndexError or silently apply sample 0's
            # grids to everyone. Fail loudly -- multi-reference batching needs ref-count bucketing.
            n_ref0 = len(samples[0]["refs"])
            if any(len(s["refs"]) != n_ref0 for s in samples):
                raise RuntimeError(
                    f"batch mixes different reference counts {sorted({len(s['refs']) for s in samples})}: "
                    f"caches are bucketed by target grid only, so multi-reference data requires "
                    f"batch=1 (or bucketing by ref-count)")
            ref_grids = [(r["grid_h"], r["grid_w"]) for r in samples[0]["refs"]]
            if any([(r["grid_h"], r["grid_w"]) for r in s["refs"]] != ref_grids
                   for s in samples[1:]):
                raise RuntimeError(
                    "batch contains different fitted reference grids; set optim.batch=1 or bucket "
                    "samples by complete reference layout")
            refs = [torch.stack([s["refs"][i]["tokens"] for s in samples]).to(device, torch.float32)
                    for i in range(len(ref_grids))]
        ref_geometry = cache_edit_ref_geometry(samples, cfg.data.edit_ref_geometry)
        ctx_p = mask_p = None
        if dop and dop_on:
            if dop_fixed is not None:
                # One encoding shared by the batch: the prior prompt does not vary per sample.
                ctx_p = dop_fixed[0].expand(len(samples), *dop_fixed[0].shape[1:])
                mask_p = dop_fixed[1].expand(len(samples), *dop_fixed[1].shape[1:])
            else:
                prior_caps = [strip_trigger(c, dp.trigger) for c in caps]
                with torch.no_grad(), (lora_disabled(dop_encoder.qwen) if te_adapters
                                       else nullcontext()):
                    ctx_p, mask_p = dop_encoder(prior_caps, images=vlm_imgs)
                ctx_p, mask_p = ctx_p.to(device).detach(), mask_p.to(device).detach()
        return (z0, ctx, mask, gh, gw, refs, ref_grids, ref_drop, delta_mask_flags,
                ref_geometry == "fit", ctx_p, mask_p)

    def sample_batch(dop=False):
        key = rng.choices(bucket_keys, weights=bucket_weights, k=1)[0]
        files = train_by_bucket[key]
        return collate([rng.choice(files) for _ in range(o.batch)],
                       drop_refs=True, augment_order=True, dop=dop)

    # ----- EMA over the adapter tensors (cheap: only the LoRA params) -----
    ema = None
    if o.use_ema:
        # Variant-agnostic: LoKr/OFT/BOFT have no lora_A/lora_B at all, and DoRA's magnitude plus
        # LoHa's second pair must be averaged too (see lora.ema_named_tensors).
        named = ema_named_tensors({"dit": adapters, "te": te_adapters})
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
                # Validation is a plain MSE against the data, never a preservation term.
                (z0, ctx, mask, gh, gw, refs, ref_grids, _, _, center_refs, _, _), vsched = packed
                t = float(vsched(torch.tensor([float(q)])).item())
                g = torch.Generator(device=device).manual_seed(cfg.runtime.seed + int(round(q * 1000)))
                if refs is not None:
                    return edit_training_step(dit, z0=z0, refs=refs, ref_grids=ref_grids, context=ctx,
                                              text_mask=mask, grid_h=gh, grid_w=gw, schedule=vsched,
                                              flow_cfg=fl, generator=g, cfg_dropout_prob=0.0,
                                              ref_dropout_prob=0.0, t_override=t,
                                              disable_weighting=True, center_refs=center_refs)
                return t2i_training_step(dit, z0=z0, context=ctx, text_mask=mask, grid_h=gh, grid_w=gw,
                                         schedule=vsched, flow_cfg=fl, generator=g, cfg_dropout_prob=0.0,
                                         t_override=t, disable_weighting=True)

            v = compute_val_loss(loss_fn, batches)
        if was_training:
            dit.train()
        return v

    def _save_adapters(path, *, ema_tag=False):
        meta = {
            "variant": lc.variant,
            "target_txtfusion": int(lc.target_txtfusion),
            "target_txtmlp": int(lc.target_txtmlp),
        }
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
    # Step-0 baseline: render the untrained base model's samples so every later sheet is comparable.
    if lg.sample_every and start_step == 0:
        do_preview("step000000_base")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # peak_gb = training peak, not the one-time model-load spike
    t0 = time.time()
    run_loss = 0.0
    run_dop = 0.0
    n_dop = 0
    n_skipped = 0
    for step in range(start_step, steps):
        _state["step"] = step
        dop_step = dop_on and (step % dp.every == 0)
        (z0, ctx, mask, gh, gw, refs, ref_grids, ref_drop, delta_mask_flags,
         center_refs, ctx_p, mask_p) = sample_batch(dop_step)
        parts = {} if dop_step else None
        schedule = get_schedule_for_seqlen(gh * gw, sigma=fl.sigma, base_shift=fl.base_shift,
                                           max_shift=fl.max_shift, base_seq_len=fl.base_image_seq_len,
                                           max_seq_len=fl.max_image_seq_len)
        if refs is not None:
            lmask = None
            if (cfg.data.masked_loss and len(refs) == 1
                    and refs[0].shape[1] == z0.shape[1]):
                lmask = reference_delta_loss_mask(
                    refs[0], z0, delta_mask_flags,
                    quantile=cfg.data.mask_quantile,
                    background_weight=cfg.data.mask_bg_weight)
            loss = edit_training_step(dit, z0=z0, refs=refs, ref_grids=ref_grids, context=ctx,
                                      text_mask=mask, grid_h=gh, grid_w=gw, schedule=schedule,
                                      flow_cfg=fl, generator=gen, cfg_dropout_prob=o.cfg_dropout_prob,
                                      ref_dropout_prob=cfg.data.ref_dropout_prob, loss_mask=lmask,
                                      ref_drop_mask=ref_drop, center_refs=center_refs,
                                      preserve_context=ctx_p, preserve_text_mask=mask_p,
                                      preserve_weight=dp.weight, loss_parts=parts)
        else:
            loss = t2i_training_step(dit, z0=z0, context=ctx, text_mask=mask, grid_h=gh, grid_w=gw,
                                     schedule=schedule, flow_cfg=fl, generator=gen,
                                     cfg_dropout_prob=o.cfg_dropout_prob,
                                     preserve_context=ctx_p, preserve_text_mask=mask_p,
                                     preserve_weight=dp.weight, loss_parts=parts)
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
        if parts:
            run_dop += parts["loss_dop"]
            n_dop += 1
        if ema is not None:
            ema.update(step)

        if (step + 1) % lg.log_every == 0:
            dt = (time.time() - t0) / lg.log_every
            rec = {"step": step + 1, "loss": run_loss / lg.log_every, "lr": sched_lr.get_last_lr()[0],
                   "s_per_step": round(dt, 3), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "skipped": n_skipped}
            if n_dop:
                # Averaged over the steps that COMPUTED it (dop.every > 1 leaves the others out),
                # and reported apart from `loss` so the regulariser cannot be mistaken for progress.
                rec["dop"] = round(run_dop / n_dop, 6)
                run_dop, n_dop = 0.0, 0
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
        if lg.sample_every and (step + 1) % lg.sample_every == 0:
            do_preview(f"step{step + 1:06d}")

    if args.smoke:
        checkpoint(steps, tag="smoke")
        do_preview("smoke")
        print("SMOKE OK", flush=True)
    else:
        checkpoint(steps, tag="final")
    tracker.close()


if __name__ == "__main__":
    main()
