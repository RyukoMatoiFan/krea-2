# Krea 2 — training

A full fine-tune / LoRA training stack for Krea 2, on top of the krea-ai/krea-2 model code
(`mmdit.py`, `encoder.py`, `autoencoder.py`, `sampling.py`), which are unchanged except a small
marked `gradient_checkpointing` hook in `mmdit.py`. Train on **Raw**, run on **Turbo** (Krea's
recommended workflow).

## Environment

Needs torch ≥2.9 (cu128), `transformers` (Qwen3-VL), `diffusers` ≥0.35 (provides
`AutoencoderKLQwenImage`), plus `safetensors einops pillow huggingface_hub pyyaml tensorboard`.
Weights are pulled from the HF cache (`HF_HOME`):
`krea/Krea-2-Raw` (`raw.safetensors`, the bf16 DiT), `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen-Image` (vae).

## Config

Typed YAML resolved as: dataclass defaults → preset (`--config`) → `user/local.yaml` (machine paths,
kept private by the `user/` gitignore) → `KREA2_<SECTION>__<KEY>` env overrides. See `training_config.py`.
Copy `config/local.example.yaml` → `user/local.yaml` and set `paths.{data_root,cache_dir,output_dir}`.

## 1. Precache (latents + text)

Encoder-only pass: VAE-encode each image → normalized, patchified latent tokens; optionally encode
the caption through the frozen Qwen3-VL (12-layer tap). One `<idx>.pt` per image.

```bash
python precache_t2i.py --config config/precache_t2i.yaml            # latents + cached text
python precache_t2i.py --config config/precache_t2i.yaml --no-cache-text   # joint trainer (live TE)
```
Captions: `<name>.txt`, or `<name>.json` read verbatim when `data.prebuilt_json: true` (structured
grounding JSON). Multi-GPU: `--num-shards N --shard K` (one process per GPU).

## 2. Full fine-tune

```bash
CUDA_VISIBLE_DEVICES=0 python train_t2i_full_joint_cached.py --config config/t2i_full.yaml [--smoke]
```
Single-GPU. The 12B DiT (+ optional Qwen3-VL TE) fits one 80GB H100 via the memory recipe in
`fused_adamw.py`: fused per-parameter backward (`accum: 1`) + on-GPU Adafactor (`optimizer_state:
adafactor`) + gradient checkpointing + bf16 stochastic rounding. `--smoke` runs ~20 steps + one
preview then exits (`SMOKE OK`). DiT-only when caches hold `llm_text` and `te_lr: 0`; set `te_lr`
(e.g. `1e-6`) to jointly fine-tune the text encoder.

Flow-matching convention (see `constants.py`): **t=1 noise, t=0 data**, `x_t = t·noise + (1-t)·x0`,
velocity target `v = noise - x0`. Timesteps use Krea's resolution-aware `mu` shift (`scheduler.py`).

## 2b. LoRA

```bash
CUDA_VISIBLE_DEVICES=0 python train_t2i_lora_cached.py --config config/t2i_lora.yaml [--smoke]
```
Base DiT frozen bf16; only the injected adapters train (`lora.py`, standard AdamW). Adapters save in
ai-toolkit/ComfyUI key format (`diffusion_model.<path>.lora_{A,B}.weight`) so a LoRA trained on Raw
loads on Turbo. Tune `lora.rank` / `lora.alpha`; `lora.target_txtfusion: true` also adapts the
text-fusion stage. Targets the attention `wq/wk/wv/wo/gate` + MLP `gate/up/down` per block.

## Resume, EMA, validation

Both trainers checkpoint weights + optimizer/scheduler/RNG and write a `resume.json` marker. Set
`paths.resume_from: auto` to continue a crashed run from the latest checkpoint; a SIGTERM/SIGINT saves
a resumable checkpoint before exit. `optim.use_ema: true` keeps a CPU EMA of the trained weights
(zero extra VRAM; saved alongside each checkpoint). `logging.val_every: N` logs a deterministic,
low-variance held-out flow-matching loss on the `data.n_eval_holdout` eval split.

## Lower VRAM: block swap

`optim.blocks_to_swap: N` parks the **N deepest** transformer blocks on CPU and pages each to the GPU
only for its forward/backward (`SingleStreamDiT.enable_block_swap`). It pairs with gradient
checkpointing — the backward recompute re-pages a block in exactly when needed — and trades VRAM for
host↔device copies (slower). Primary use is the **LoRA** trainer: only the frozen base is paged, so
adapters stay resident and trainable. The full-FT trainer also supports it (it must page trainable
weights too — noticeably slower; full-FT already fits one 80GB GPU, so leave it `0` there unless
pushing resolution/batch). At 1024 with 14/28 blocks swapped, a LoRA run drops peak VRAM
**~31 → 20 GB** (−36%) for ~1.4× the per-step time; training dynamics are unchanged.

```bash
KREA2_OPTIM__BLOCKS_TO_SWAP=14 python train_t2i_lora_cached.py --config config/t2i_lora.yaml
```

## 3. Monitor

Metrics stream to `<output_dir>/metrics.jsonl` (loss, lr, s/step, peak VRAM) and tensorboard; preview
contact-sheets land in `<output_dir>/samples/`. Live dashboard:

```bash
python dashboard.py --run <output_dir>
```

## 4. Edit / multi-reference

`precache_edit.py` caches a target + reference image(s) + instruction; the trainers auto-detect the
`refs` field in the cache and switch to `edit_training_step` (packs `[text, refs(clean), target(noised)]`,
loss on the target only) — no separate trainer.

```bash
python precache_edit.py --config config/precache_t2i.yaml --manifest meta.jsonl --data-root /data/edit
python train_t2i_lora_cached.py --config config/t2i_lora.yaml      # refs auto-detected -> edit step
python sample_edit.py --config config/t2i_lora.yaml --lora runs/r/ckpts/lora_final.safetensors \
    --prompt "make it autumn" --ref source.jpg --out edited.png    # repeat --ref for multiref
```
Manifest line: `{"target": "...", "refs": ["..."], "caption"|"instruction": "..."}`. Edit training needs
paired (source→target, instruction) data; structural edits (object removal, background replacement)
favour full fine-tune over LoRA.

## 5. Style transfer

* **In-context style reference (reuses the edit path):** the style image is the reference and the
  caption describes the target *content*; train with `precache_edit` + the edit trainer
  (`data.ref_dropout_prob` gives a style-strength CFG knob). Uses same-style/different-content pairs.
* **Native image conditioning (Qwen3-VL):** `encoder.py` feeds the style image through the VLM so
  image-derived tokens enter the `(B,L,12,2560)` stream. `precache_style.py` caches that image-conditioned
  `llm_text`, so the standard DiT-only trainer consumes it with **no trainer change**; inference is
  `sample.py --style-ref <image>`.

```bash
python precache_style.py --config config/precache_t2i.yaml --manifest style.jsonl --data-root /data/style
python train_t2i_lora_cached.py --config config/t2i_lora.yaml      # learns to use image conditioning
python sample.py --config config/t2i_lora.yaml --lora runs/r/ckpts/lora_final.safetensors \
    --style-ref style.jpg --prompt "a city street" --out styled.png
```
Style manifest: `{"target": "...", "style": "<style image>", "caption": "..."}`. The base model is
text-only, so the DiT must be (LoRA-)trained to consume image conditioning; this uses same-style/
different-content training pairs.

## Inference recipe (train on Raw, run on Turbo)

`sample.py` uses the reference sampler. Raw: `--steps 52 --guidance 3.5`. **Turbo** (run a Raw-trained
LoRA on the distilled checkpoint): `--base krea/Krea-2-Turbo --base-file turbo.safetensors --steps 8
--guidance 0 --mu 1.15 --lora <your_lora>.safetensors`.
