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
Single-GPU training uses the memory recipe in `fused_adamw.py`: fused per-parameter backward
(`accum: 1`) + on-GPU Adafactor (`optimizer_state:
adafactor`) + gradient checkpointing + bf16 stochastic rounding. `--smoke` runs ~20 steps + one
preview then exits (`SMOKE OK`). DiT-only when caches hold `llm_text` and `te_lr: 0`; set `te_lr`
(e.g. `1e-6`) to jointly fine-tune the text encoder.

Throughput switches, all off by default and none changing the objective:

| option | effect |
|---|---|
| `optim.adafactor_kernel: triton` | same update without materialising the fp32 gradient copy or the update tensor |
| `data.pad_to_multiple: 0` | no sequence padding; with one sample per step the mask is empty and attention runs unmasked |
| `data.skip_ref_cross_attention: true` | multi-reference only: block-sparse mask dropping attention between *different* references (exact — those blocks carry no meaning) |

Flow-matching convention (see `constants.py`): **t=1 noise, t=0 data**, `x_t = t·noise + (1-t)·x0`,
velocity target `v = noise - x0`. Timesteps normally use Krea's resolution-aware `mu` shift
(`scheduler.py`). `flow.timestep_weighting: weighted` selects the empirical 1,000-step recipe:
uniform unshifted timestep sampling plus its mean-one lookup-table loss weights.

Reference-conditioned cache behaviour is declared, never inferred from filenames. `precache_edit.py`
manifest rows accept two optional fields: `reference_delta_mask: true` (first reference is spatially
aligned with the target → delta-based loss mask) and `validation_group` (subgroup metrics label).
Existing caches supply the same fields through `data.cache_annotations`, keyed by cache filename:

```json
{
  "example.pt": {
    "reference_delta_mask": true,
    "validation_group": "primary"
  }
}
```

## 2b. LoRA

```bash
CUDA_VISIBLE_DEVICES=0 python train_t2i_lora_cached.py --config config/t2i_lora.yaml [--smoke]
```
Base DiT frozen bf16; only the injected adapters train (`lora.py`, standard AdamW). Adapters save in
the ComfyUI key format (`diffusion_model.<path>.lora_{A,B}.weight`) so a LoRA trained on Raw
loads on Turbo. Tune `lora.rank` / `lora.alpha`; `lora.target_txtfusion` and `lora.target_txtmlp`
default on to adapt the text-fusion and text-MLP stages. Targets the attention `wq/wk/wv/wo/gate`
and MLP `gate/up/down` per block, plus the `txtmlp` projection Linears.

`lora.variant` selects the update parameterisation. All six share the targets, save/load format and
runtime scale knob, and are exact no-ops at step 0.

| `variant` | update | notes |
|---|---|---|
| `lora` | `dW = B A`, low-rank product | the default |
| `dora` | weight split into magnitude and direction, LoRA on the direction | lets direction change without forcing a change of scale |
| `loha` | `dW = (B1 A1) * (B2 A2)`, elementwise | higher effective rank for the same parameter count |
| `lokr` | `dW = A (x) B`, Kronecker product | ranks multiply rather than add; smallest files |
| `oft` | `W' = R W`, `R` block-diagonal orthogonal (Cayley) | preserves neuron angles and norms by construction |
| `boft` | `R` factorised into butterfly stages | same property, parameter cost linear in the width |

`loha` and `lokr` are the LyCORIS algorithms. For `oft` / `boft`, `lora.rank` is the **number of
blocks**: larger means *fewer* parameters. Every variant except `lora` changes the rank/parameter
relationship, so compare them at equal parameter count, not equal rank.

A *small* `rank` is the expensive end for `oft`/`boft`, which is the opposite of the LoRA intuition:
few blocks means wide blocks, and the rotation costs `tokens × out_features × block_width`, in both
time and peak memory. A low block count can therefore make the rotation, not the model, the dominant
consumer — while a high one keeps it marginal. Start high (32–64) and only lower it if the adapter is
underfitting.

The variant also applies to TE-LoRA (`lora.te_rank`, both trainers) and to the concept slider. It is
recorded in the saved file's metadata and read back on load, so `sample.py --lora`,
`sample_edit.py`, `eval_t2i.py` and `slider_render.py` reconstruct the right adapter — including
`alpha`, which scales the adapter's effect and is not assumed equal to `rank`. `optim.use_ema`
averages every tensor of whichever variant is active.

For a fresh higher-resolution stage, set `paths.lora_init` to an earlier LoRA checkpoint. This loads
adapter weights only and starts at step 0 with a new optimizer, LR schedule, and RNG. A normal
`paths.resume_from` takes precedence and restores the complete interrupted training state instead.

### Differential output preservation

A LoRA taught a new concept also drags everything near it: the trigger bleeds into unrelated prompts
and the class it belongs to collapses toward the training set. The classic fix is a regularisation
image set — generate hundreds of "a photo of a dog" images and train on them alongside. Differential
output preservation removes that step. Every step it also asks the model to predict a **prior prompt**,
and supervises that prediction against **the frozen base's own prediction of the same prompt** —
obtained by running the same forward with the adapters switched off (`lora_disabled`). The prior is
therefore exact rather than approximated by generated images, and the term is **identically zero at
step 0**, growing only as the adapter changes behaviour on text the training set is not about.

```yaml
dop:
  enabled: true
  weight: 1.0
  prompt: "a photo of a dog"   # fixed prior prompt, encoded ONCE at startup
  # trigger: "sks"             # OR: prior = each caption with the trigger word removed
  every: 1
```

Set exactly one of `prompt` or `trigger`. `prompt` is the cheap default. `trigger` is the tighter
constraint — the prior becomes the *same sentence without the concept being taught* — but costs a
live text encode of a second prompt every step, and needs captions in the cache.

- **Costs roughly a doubled step, and raises peak memory**: one extra forward with gradients and one
  without. Both the main and the preservation graph stay alive until backward, so activation memory
  rises as well as time — budget for it before enabling this on a configuration that is already near
  the memory ceiling. `every: N` amortises the time, but applies the regulariser unevenly; prefer
  lowering `weight`.
- Logged as a separate `dop` field in `metrics.jsonl`, never folded into `loss`, so the regulariser
  cannot be mistaken for training progress. It should rise off zero and then stay bounded; a `dop`
  that climbs steadily means the adapter is drifting further from the prior every step.
- It shares the noised latent, timestep and reference dropout with the main loss, and the same
  timestep weighting, so `weight` is directly comparable to the flow loss. It applies no edit
  mask — preservation is global by definition.
- **Adapters only.** With `lora.train_transformer: false` the term is identically zero and is
  rejected. A full fine-tune would need a frozen reference copy of the model; this uses the fact
  that an adapted model contains its own base.
- With TE-LoRA the prior prompt is encoded with the **TE adapters disabled**, so the prior
  conditioning is a constant of the base model; the term then constrains the DiT.
- With `optim.compile_blocks` each block compiles twice (adapters on and off) — a one-off cost.

## 2c. Concept sliders

`train_slider.py` trains a bidirectional attribute LoRA with **no dataset**: it regresses a ±adapter
onto the frozen base's own velocity, nudged along (`slider.positive` − `slider.negative`) at a neutral
anchor. Context latents are self-generated rollouts (or `slider.rollouts: 0` reads `paths.data_root`
for a domain-specific knob). Set the axis in `config/slider.yaml` or via env:

```bash
KREA2_SLIDER__POSITIVE="sharp, 4K, crisp, fine detail" KREA2_SLIDER__NEGATIVE="blurry, soft, low detail" \
  python train_slider.py --config config/slider.yaml
```
Dial the knob at inference with `sample.py --lora-scale` (`1.0` normal, `0` off, `<0` inverts) — positive
→ the attribute, negative → away from it. `slider_render.py` renders a multi-scale `[-s | off | +s]`
sweep. Subtle high-frequency axes (detail/sharpness) need a higher `slider.eta` than strong global ones
(exposure); `slider.late_frac < 1` confines training to the low-noise (texture) steps.

## Resume, EMA, validation

Both trainers checkpoint weights + optimizer/scheduler/RNG and write a `resume.json` marker.

- `paths.resume_from: auto` — continue from the latest checkpoint. SIGTERM/SIGINT saves a resumable
  checkpoint before exit.
- `optim.use_ema: true` — CPU EMA of the trained weights, no extra VRAM, saved with each checkpoint.
- `logging.val_every: N` — deterministic held-out flow-matching loss on the `data.n_eval_holdout` split.

## Lower VRAM: block swap + base quantization

Three independent levers. They compose, and none changes the training objective.

| option | stores / does | requires |
|---|---|---|
| `optim.blocks_to_swap: N` | parks the N deepest blocks on CPU, pages each in for its forward/backward | gradient checkpointing; costs host↔device copies |
| `optim.quantize_base: fp8` | frozen attn+MLP weights as e4m3 + per-row scale, dequantized per forward | gradient checkpointing (forced on) |
| `optim.quantize_base: int8` | the same weights rotated and stored as int8; the **matmul runs on int8 tensor cores**, so compute drops too | int8 tensor cores (Ampere+); incompatible with `variant: dora` |
| `optim.quantize_te: fp8` | frozen live Qwen3-VL linear weights as e4m3 + per-row scale | frozen TE only; incompatible with TE full fine-tuning and TE-LoRA |
| `optim.compile_blocks: true` | compiles each DiT block after adapter injection | LoRA transformer training; incompatible with block swap |

```bash
KREA2_OPTIM__BLOCKS_TO_SWAP=14 python train_t2i_lora_cached.py --config config/t2i_lora.yaml
KREA2_OPTIM__QUANTIZE_BASE=fp8 KREA2_OPTIM__BLOCKS_TO_SWAP=14 python train_t2i_lora_cached.py --config config/t2i_lora.yaml
KREA2_OPTIM__QUANTIZE_BASE=int8 python train_t2i_lora_cached.py --config config/t2i_lora.yaml
KREA2_OPTIM__QUANTIZE_TE=fp8 python train_t2i_lora_cached.py --config config/t2i_lora.yaml
KREA2_OPTIM__COMPILE_BLOCKS=true python train_t2i_lora_cached.py --config config/t2i_lora.yaml
```

- `quantize_base` is LoRA-only. Block swap also runs in full-FT, paging trainable weights.
- Block swap keeps adapters resident (`skip_trainable=True`); the backward recompute re-pages a block
  exactly when needed.
- fp8 forces gradient checkpointing on: otherwise the per-forward dequant is retained across every
  layer and the saving is lost.
- int8 (`convrot_int8.py`) rotates weights and activations by a fixed block Hadamard so one scale per
  row stays accurate. The rotation is folded into the weight offline and cancels inside the matmul.
  Backward is a straight-through estimator producing only an input gradient — hence frozen base with
  adapters, never full fine-tune. `dora` needs a dequantized weight and refuses.

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

Edit geometry and grounded-text controls:

| option | behavior |
|---|---|
| `data.aspect_bucketing: true` | buckets edit targets by native aspect ratio during precache |
| `data.edit_ref_geometry: fit` | fits each reference inside the target grid without aspect squash and fractionally centers its RoPE coordinates |
| `data.fit_ref_crop_tolerance: 0.08` | near-matching reference aspect ratios within 8% minimally crop to fill the target |
| `data.edit_vlm_cond: true` | sends the original reference pixels through Qwen3-VL as well as the VAE-token path |
| `data.vlm_image_size: 768` + `data.vlm_image_min_size: 384` | jitters the Qwen3-VL grounding cap per training step; previews and inference remain fixed at the maximum |

`edit_vlm_cond` requires `precache_edit.py --no-cache-text`; a cached text tensor cannot represent
per-step image grounding. Fit caches record their geometry and are refused if the training config
does not match. Fitted references with different token grids require `optim.batch: 1`; use
`optim.accum` for a larger effective batch.

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
--guidance 0 --mu 1.15 --lora <your_lora>.safetensors`. Add `--lora-scale <s>` to dial a concept
slider.

## Regional prompting

`sampling.sample_regions` places a different prompt in each image region. Opt-in — the default
sampler is byte-identical. Each region's image tokens are routed to their own text segment via the
model's `attn_mask_override` / `txt_attn_override` kwargs (both default `None`).

- `regions=[{"prompt": ..., "box": (x0, y0, x1, y1)}]`, box edges as 0–1 fractions.
- `isolate_regions` (default on) keeps each region's image self-attention local: distinct placed
  subjects, at the cost of a seam. Drop it for one coherent scene.
- Combines with a LoRA to place trained identities or styles per region.
