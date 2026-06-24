"""Typed YAML config for the Krea 2 training scripts.

Keeps all machine-specific paths and run hyperparameters out of code. Scripts do
``from krea2.training_config import load_config, apply_runtime`` and read everything from the
returned :class:`TrainConfig`.

Resolution order (last wins):
  1. dataclass defaults
  2. the YAML at the given path (or ``$KREA2_CONFIG`` when path is None)
  3. ``user/local.yaml`` deep-merged if present -- holds real machine paths, kept private
     by the ``user/`` gitignore entry
  4. ``KREA2_<SECTION>__<KEY>`` env overrides (e.g. ``KREA2_PATHS__DIT_REPO``,
     ``KREA2_OPTIM__STEPS``), cast to the target field type.

Only stdlib + PyYAML are imported at module load; ``torch`` is imported lazily inside
:func:`dtype_of` / :func:`set_tf32`, so this module can be imported without torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

import yaml

from constants import TEXT_ENCODER_ID, VAE_ID

_ENV_PREFIX = "KREA2_"


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
  # The DiT (SingleStreamDiT) weights: a single safetensors in an HF repo (or a local dir +
  # filename). The custom fork loads this directly via load_file; encoder + VAE load from HF ids.
  dit_repo: str = "krea/Krea-2-Raw"     # HF repo id OR a local dir
  dit_file: str = "raw.safetensors"     # weight file within dit_repo
  text_encoder_id: str = TEXT_ENCODER_ID  # Qwen3-VL-4B-Instruct
  vae_id: str = VAE_ID                   # Qwen/Qwen-Image (subfolder=vae)
  data_root: str = "data"
  cache_dir: str = ""        # "" -> f"{data_root}/cache"
  output_dir: str = "runs"
  ckpt_dir: str = ""         # "" -> f"{output_dir}/ckpts"
  results_dir: str = ""
  resume_from: str = ""      # crash recovery: "auto" = resume the same run from {ckpt_dir} marker


@dataclass
class RuntimeConfig:
  device: str = "cuda"
  dtype: str = "bfloat16"
  hf_offline: bool = False
  extra_sys_path: list = field(default_factory=list)
  seed: int = 0
  tf32: bool = True           # enable TF32 matmul/cudnn (free speedup on Ampere+/Hopper)


@dataclass
class LoraConfig:
  rank: int = 64
  alpha: float | None = None
  variant: str = "lora"        # lora | dora | loha | lokr
  target_txtfusion: bool = False  # also adapt the text-fusion stage Linears
  te_rank: int = 0             # TE-LoRA rank; 0 -> use rank
  train_transformer: bool = True  # TE trainer: also LoRA the DiT (False = DiT frozen, TE-only)


@dataclass
class DataConfig:
  resolution: int = 1024
  img_ext: str = "jpg"
  instr_field: str = "edit"        # (edit/multiref) field naming the instruction in metadata
  meta_at_root: bool = False
  limit: int = 0
  n_eval_holdout: int = 4
  prebuilt_json: bool = False      # read a verbatim JSON caption sidecar per image
  json_suffix: str = ".json"
  masked_loss: bool = False        # (edit) weight loss to the edited region
  mask_quantile: float = 0.5
  mask_bg_weight: float = 0.0
  ref_dropout_prob: float = 0.0    # (edit/multiref/style) per-sample reference dropout for CFG on refs
  train_list: str = ""             # JSON list of cache filenames (repeats = oversampling)
  aspect_bucketing: bool = False   # precache at nearest-AR bucket instead of square-squash
  bucket_pixels: int = 0           # target area for buckets; 0 -> resolution^2
  num_buckets: int = 9


@dataclass
class OptimConfig:
  lr: float = 1e-5
  steps: int = 40000
  batch: int = 1
  accum: int = 1
  warmup: int = 200
  optimizer: str = "adamw"   # LoRA: adamw | adamw8bit | prodigy | schedule_free | came
  grad_clip: float = 1.0
  grad_checkpointing: bool = True
  cfg_dropout_prob: float = 0.1
  use_ema: bool = False            # maintain an EMA of the trained weights (LoRA adapter or full DiT)
  ema_decay: float = 0.999
  ema_every: int = 10              # stride for the (CPU) EMA host-copy; decay is compounded as decay**every
  nan_guard: bool = True           # skip the optimizer step on a non-finite loss
  lr_scheduler: str = "cosine"     # cosine | constant | linear | cosine_restarts
  min_lr_ratio: float = 0.1        # LR floor (fraction of base) the decay approaches
  num_restarts: int = 1            # cycles for cosine_restarts
  offload_optimizer: bool = False  # full-FT: keep Adam moments in CPU RAM (lower VRAM, host-RAM cost)
  blocks_to_swap: int = 0          # page N deepest blocks CPU<->GPU per fwd/bwd (LoRA: frozen base only; full-FT pages all, slower)
  quantize_base: str = ""          # LoRA: "" off | "fp8" -> e4m3-quantize the frozen base blocks (~half base VRAM)
  weight_decay: float = 0.01       # full-FT AdamW weight decay
  te_lr: float = 0.0               # joint full-FT: separate (lower) LR for the text encoder;
                                   # 0 -> lr/10 when training the DiT too, else lr (TE-only stage)
  train_dit: bool = True           # joint trainer: also full-FT the DiT. False = DiT frozen, TE-only
  optimizer_state: str = "adafactor"  # full-FT moment backend: adamw (offload) | adafactor (on-GPU)
  preload_caches: bool = True      # full-FT trainers: preload latent caches into RAM


@dataclass
class FlowConfig:
  # Krea 2 resolution-aware ("dynamic") timestep shift. mu is linearly interpolated by IMAGE-TOKEN
  # count between (base_image_seq_len, base_shift) and (max_image_seq_len, max_shift), then applied
  # as ts = exp(mu)/(exp(mu)+(1/ts-1)**sigma). Training draws t = schedule(rand); sampling maps the
  # Euler grid. Defaults are the diffusers Krea2 pipeline values.
  base_shift: float = 0.5
  max_shift: float = 1.15
  base_image_seq_len: int = 256
  max_image_seq_len: int = 6400
  sigma: float = 1.0
  timestep_weighting: str = "uniform"  # uniform | bell | min_snr | sigma_sqrt | cosmap
  min_snr_gamma: float = 5.0
  noise_offset: float = 0.0            # per-channel constant added to the sampled noise
  input_perturbation: float = 0.0      # extra noise on x_t only (target stays clean)


@dataclass
class LoggingConfig:
  log_every: int = 25
  ckpt_every: int = 1000
  val_every: int = 0          # 0 = off; else held-out validation loss every N steps
  sample_every: int = 0       # 0 = off; else decode in-training samples every N steps
  sample_steps: int = 28      # sampler steps for in-training samples (Krea Base default)
  sample_guidance: float = 4.5
  sample_count: int = 4       # how many prompts/items to sample each time
  edit_preview_manifest: str = ""  # edit runs: JSON list of {src,instruction,tgt}; renders [src|edit|tgt] rows + a step-0 base sheet
  tracker: str = "tensorboard"  # none | wandb | tensorboard (mirrors metrics.jsonl scalars)
  wandb_project: str = "krea2"
  run_name: str = ""          # tracker run name ("" -> backend default)


@dataclass
class SliderConfig:
  # Concept-Slider LoRA (train_slider.py): a bidirectional attribute knob in velocity space.
  positive: str = ""           # c+ : attribute pushed at +scale (e.g. "dark, dim, low-key")
  negative: str = ""           # c- : attribute pushed at -scale (e.g. "bright, well-lit")
  anchor: str = ""             # conditioning the slider modifies; "" -> unconditional
  eta: float = 2.0             # direction strength defining the train targets
  train_scale: float = 1.0     # +/- adapter scale trained at
  infer_scale: float = 1.0     # default inference strength (lora_scaled factor)
  bidirectional: bool = True   # True: ± knob; False: enhance-only (+ branch)
  late_frac: float = 1.0       # <1 restricts training to low-noise (late/detail) steps t in (0, late_frac)
  rollouts: int = 8            # self-generate N context images; 0 -> read paths.data_root folder
  eval_scales: str = "0.6,1.0,1.4"  # comma list of inference scales rendered in the [-s|off|+s] sweep


@dataclass
class TrainConfig:
  paths: PathsConfig = field(default_factory=PathsConfig)
  runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
  lora: LoraConfig = field(default_factory=LoraConfig)
  data: DataConfig = field(default_factory=DataConfig)
  optim: OptimConfig = field(default_factory=OptimConfig)
  flow: FlowConfig = field(default_factory=FlowConfig)
  logging: LoggingConfig = field(default_factory=LoggingConfig)
  slider: SliderConfig = field(default_factory=SliderConfig)


# Section name -> dataclass type. Drives merging and env-override casting.
_SECTIONS = {
  "paths": PathsConfig,
  "runtime": RuntimeConfig,
  "lora": LoraConfig,
  "data": DataConfig,
  "optim": OptimConfig,
  "flow": FlowConfig,
  "logging": LoggingConfig,
  "slider": SliderConfig,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
  """Recursively merge ``override`` into ``base``, returning a new dict."""
  out = dict(base)
  for key, val in override.items():
    if isinstance(val, dict) and isinstance(out.get(key), dict):
      out[key] = _deep_merge(out[key], val)
    else:
      out[key] = val
  return out


def _load_yaml(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if data is None:
    return {}
  if not isinstance(data, dict):
    raise ValueError(f"config {path} must be a mapping at the top level")
  return data


def _check_unknown_keys(data: dict, path: str) -> None:
  """Raise a clear error for typo'd / unknown section or field names."""
  for section, values in data.items():
    if section not in _SECTIONS:
      raise ValueError(
        f"unknown config section {section!r} in {path}; valid sections: {sorted(_SECTIONS)}"
      )
    if values is None:
      continue
    if not isinstance(values, dict):
      raise ValueError(
        f"section {section!r} in {path} must be a mapping, got {type(values).__name__}"
      )
    valid = {f.name for f in fields(_SECTIONS[section])}
    for key in values:
      if key not in valid:
        raise ValueError(
          f"unknown key {section}.{key!r} in {path}; valid keys: {sorted(valid)}"
        )


def _cast_to(field_type, raw: str):
  """Cast a string env value to a dataclass field's annotated type."""
  type_str = str(field_type)
  if field_type is bool or "bool" in type_str:
    return raw.strip().lower() in ("1", "true", "yes")
  if field_type is int or "int" in type_str:
    return int(raw)
  if field_type is float or "float" in type_str:
    return float(raw)
  return raw


def _apply_env_overrides(data: dict) -> dict:
  """Apply ``KREA2_<SECTION>__<KEY>=value`` env vars onto the merged dict."""
  out = dict(data)
  for env_name, raw in os.environ.items():
    if not env_name.startswith(_ENV_PREFIX) or "__" not in env_name:
      continue
    body = env_name[len(_ENV_PREFIX):]
    section_part, key_part = body.split("__", 1)
    section = section_part.lower()
    key = key_part.lower()
    if section not in _SECTIONS:
      continue
    field_map = {f.name: f.type for f in fields(_SECTIONS[section])}
    if key not in field_map:
      continue
    out.setdefault(section, {})
    out[section][key] = _cast_to(field_map[key], raw)
  return out


def _build_section(cls, values: dict):
  """Instantiate a section dataclass from its (already validated) dict."""
  return cls(**(values or {}))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_config(path: str | None = None) -> TrainConfig:
  """Build a :class:`TrainConfig` from defaults, YAML, local.yaml, and env.

  ``path`` (or ``$KREA2_CONFIG`` when None) names the preset YAML. A non-None path that does
  not exist raises ``FileNotFoundError``. Unknown YAML keys raise ``ValueError`` (typo protection).
  """
  if path is None:
    path = os.environ.get(f"{_ENV_PREFIX}CONFIG")

  merged: dict = {}
  yaml_dir = None

  if path is not None:
    if not os.path.exists(path):
      raise FileNotFoundError(f"config file not found: {path}")
    preset = _load_yaml(path)
    _check_unknown_keys(preset, path)
    merged = _deep_merge(merged, preset)
    yaml_dir = os.path.dirname(os.path.abspath(path))

  # Deep-merge user/local.yaml if present (private machine paths; user/ is gitignored).
  local_path = os.path.join(os.getcwd(), "user", "local.yaml")
  if os.path.exists(local_path):
    local = _load_yaml(local_path)
    _check_unknown_keys(local, local_path)
    merged = _deep_merge(merged, local)

  # Env overrides last.
  merged = _apply_env_overrides(merged)

  cfg = TrainConfig(
    **{name: _build_section(cls, merged.get(name)) for name, cls in _SECTIONS.items()}
  )

  # Resolve empty-string path defaults after all merging.
  if not cfg.paths.cache_dir:
    cfg.paths.cache_dir = f"{cfg.paths.data_root}/cache"
  if not cfg.paths.ckpt_dir:
    cfg.paths.ckpt_dir = f"{cfg.paths.output_dir}/ckpts"

  return cfg


def set_tf32(enabled: bool) -> None:
  """Toggle TF32 for matmul + cuDNN."""
  import torch

  torch.backends.cuda.matmul.allow_tf32 = enabled
  torch.backends.cudnn.allow_tf32 = enabled
  torch.set_float32_matmul_precision("high" if enabled else "highest")


def apply_runtime(cfg: TrainConfig) -> None:
  """Apply runtime side effects: sys.path, offline env, TF32."""
  import sys

  for entry in cfg.runtime.extra_sys_path:
    if entry and entry not in sys.path:
      sys.path.insert(0, entry)

  if cfg.runtime.hf_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

  if cfg.runtime.tf32:
    set_tf32(True)

  # mmdit.py decorates RMSNorm / PositionalEncoding / LastLayer with @torch.compile(fullgraph=True).
  # During training (+ grad-checkpointing) and the no-grad sampler, grad-mode and sequence shapes
  # change, thrashing dynamo's recompile limit and HARD-failing previews (fullgraph=True). Compile
  # is an inference nicety we don't need for training correctness -> disable dynamo for our runs.
  os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
  try:
    import torch._dynamo
    torch._dynamo.config.disable = True
  except Exception:
    pass


def dtype_of(cfg: TrainConfig):
  """Resolve ``runtime.dtype`` to a ``torch.dtype`` (torch imported lazily)."""
  import torch

  mapping = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
  }
  name = (cfg.runtime.dtype or "bfloat16").lower()
  if name not in mapping:
    raise ValueError(f"unsupported runtime.dtype {cfg.runtime.dtype!r}")
  return mapping[name]
