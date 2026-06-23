"""Reusable training utilities: flow-matching timestep shift/weighting, masked loss,
optimizer factory (incl. Prodigy), and a NaN/inf guard.

These are model-agnostic helpers shared by the Krea 2 trainers.

Conventions (see constants.py): flow-matching with t=1 noise, t=0 data,
x_t = t*noise + (1-t)*x0, velocity target v = noise - x0; the noise fraction
(``sigma``) equals t.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Timestep shift (flux/SD3-style resolution shift), applied to sampled t.
# --------------------------------------------------------------------------- #
def apply_flux_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
  """Monotonic reparametrization of t in [0,1] with fixed endpoints.

  t' = shift * t / (1 + (shift - 1) * t).
  shift == 1 -> identity. shift > 1 biases samples toward t=1 (the noise side in
  this repo's t=1-noise convention); shift < 1 biases toward t=0 (the data side).
  Endpoints 0 and 1 are fixed for any shift > 0.
  """
  if shift == 1.0:
    return t
  if shift <= 0.0:
    raise ValueError(f"shift must be > 0, got {shift}")
  return (shift * t) / (1.0 + (shift - 1.0) * t)


# --------------------------------------------------------------------------- #
# Per-sample timestep loss weighting.
# --------------------------------------------------------------------------- #
def timestep_weight(
  t: torch.Tensor, scheme: str = "uniform", *, gamma: float = 5.0, bell_sigma: float = 0.25
) -> torch.Tensor:
  """Per-sample loss weight as a function of timestep t (shape (B,)).

  schemes (sigma = noise fraction = t in this repo's t=1-noise convention):
    "uniform"    -> 1 everywhere (no reweighting).
    "bell"       -> Gaussian centred at t=0.5 (emphasise mid-noise timesteps).
    "min_snr"    -> flow-matching min-SNR-gamma: min(SNR, gamma)/SNR, with
                    SNR(t) = ((1 - t) / t)^2. Down-weights easy (high-SNR / low-noise)
                    timesteps so they do not dominate the gradient.
    "sigma_sqrt" -> 1 / sigma^2 (SD3 loss weighting). Strongly up-weights low-noise
                    (low-t) timesteps.
    "cosmap"     -> 2 / (pi * (1 - 2*sigma + 2*sigma^2)) (SD3 cosine-map loss weighting).
  Returns weights with the same shape as t (non-negative).
  """
  if scheme == "uniform":
    return torch.ones_like(t)
  if scheme == "bell":
    return torch.exp(-((t - 0.5) ** 2) / (2.0 * bell_sigma ** 2))
  if scheme == "min_snr":
    tc = t.clamp(1e-4, 1.0 - 1e-4)
    snr = ((1.0 - tc) / tc) ** 2  # krea2: t=1 noise, so SNR = ((1-t)/t)^2
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr
  if scheme == "sigma_sqrt":
    sigma = t.clamp(1e-4, 1.0)  # krea2: noise fraction == t
    return sigma ** -2.0
  if scheme == "cosmap":
    sigma = t.clamp(0.0, 1.0)  # krea2: noise fraction == t
    bot = 1.0 - 2.0 * sigma + 2.0 * sigma ** 2  # >= 0.5, never zero
    return 2.0 / (math.pi * bot)
  raise ValueError(f"unknown timestep weighting scheme: {scheme!r}")


# --------------------------------------------------------------------------- #
# Masked + weighted flow-matching loss.
# --------------------------------------------------------------------------- #
def flow_loss(
  pred: torch.Tensor,
  target: torch.Tensor,
  *,
  mask: Optional[torch.Tensor] = None,
  weight: Optional[torch.Tensor] = None,
  eps: float = 1e-8,
) -> torch.Tensor:
  """Masked, timestep-weighted MSE over (B, n, D) velocity tensors.

  mask:   per-token weights (B, n) or per-element (B, n, D), or None. Loss is the
          mean squared error over masked elements only (so a region/edit mask
          focuses the gradient there).
  weight: per-sample weights (B,) from ``timestep_weight``, or None. Combined as a
          weighted mean over the batch.
  """
  se = (pred - target) ** 2  # (B, n, D)
  if mask is not None:
    if mask.dim() == se.dim() - 1:
      mask = mask.unsqueeze(-1)  # (B, n, 1)
    mask = mask.to(se.dtype)
    se = se * mask
    norm = mask.expand_as(se).sum(dim=(1, 2)).clamp_min(eps)  # (B,)
    per_sample = se.sum(dim=(1, 2)) / norm
  else:
    per_sample = se.mean(dim=tuple(range(1, se.dim())))  # (B,)

  if weight is not None:
    weight = weight.to(per_sample.dtype)
    return (per_sample * weight).sum() / weight.sum().clamp_min(eps)
  return per_sample.mean()


# --------------------------------------------------------------------------- #
# Aspect-ratio bucketing (train at native-ish AR instead of square-squash).
# --------------------------------------------------------------------------- #
def aspect_buckets(target_pixels: int = 512 * 512, *, divisor: int = 16,
                   min_ar: float = 0.5, max_ar: float = 2.0, num: int = 9) -> list:
  """Generate `num` (H, W) pixel buckets of ~`target_pixels` area spanning the AR
  range [min_ar, max_ar] (AR = W/H), each side a multiple of `divisor` (the VAE
  patch * ae_scale, so the latent grid is integral). Deduplicated + sorted."""
  buckets = set()
  for i in range(num):
    ar = min_ar * (max_ar / min_ar) ** (i / max(1, num - 1))  # log-spaced W/H
    h = max(divisor, round(math.sqrt(target_pixels / ar) / divisor) * divisor)
    w = max(divisor, round(h * ar / divisor) * divisor)
    buckets.add((h, w))
  return sorted(buckets)


def nearest_bucket(w: int, h: int, buckets: list) -> tuple:
  """Pick the bucket whose aspect ratio (W/H) is closest (in log-space) to the image."""
  ar = math.log(max(1, w) / max(1, h))
  return min(buckets, key=lambda b: abs(math.log(b[1] / b[0]) - ar))


def noise_with_offset(shape, offset: float, *, generator=None, device=None, dtype=torch.float32):
  """Sample (B, n, D) noise with an optional per-channel constant offset.

  Noise offset (Guttenberg) adds a per-channel constant shared across tokens, which
  lets the model learn global tone/brightness shifts. offset=0 -> plain N(0,1).
  """
  noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
  if offset:
    per_channel = torch.randn(
      (shape[0], 1, shape[-1]), generator=generator, device=device, dtype=dtype
    )
    noise = noise + offset * per_channel
  return noise


def derive_edit_mask(
  z_ref: torch.Tensor, z_tgt: torch.Tensor, *, quantile: float = 0.5
) -> torch.Tensor:
  """Per-token edit mask from |z_tgt - z_ref|: tokens that actually change.

  z_ref, z_tgt: (B, n, D) or (n, D). Returns a float mask (B, n) or (n,) with 1.0
  on tokens whose mean abs-difference is at or above the per-sample ``quantile``
  threshold, 0.0 elsewhere. Useful for edit training so the loss concentrates on
  the changed region instead of the (unchanged) background.
  """
  diff = (z_tgt - z_ref).abs().mean(dim=-1)  # (..., n)
  if diff.dim() == 1:
    thr = torch.quantile(diff, quantile)
    return (diff >= thr).to(z_ref.dtype)
  thr = torch.quantile(diff, quantile, dim=-1, keepdim=True)  # (B, 1)
  return (diff >= thr).to(z_ref.dtype)


# --------------------------------------------------------------------------- #
# LR schedulers (warmup + cosine / constant / linear / cosine-restarts).
# --------------------------------------------------------------------------- #
def build_lr_scheduler(
  optimizer,
  *,
  scheduler: str = "cosine",
  warmup: int = 0,
  total_steps: int = 1,
  num_restarts: int = 1,
  min_lr_ratio: float = 0.0,
):
  """LambdaLR with a linear warmup followed by the chosen decay.

  scheduler: cosine | constant | linear | cosine_restarts. `min_lr_ratio` is the
  floor (fraction of base LR) the decays approach.
  """
  def fn(step: int) -> float:
    if warmup and step < warmup:
      return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    if scheduler == "constant":
      return 1.0
    if scheduler == "linear":
      return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
    if scheduler == "cosine":
      return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if scheduler == "cosine_restarts":
      cyc = progress * max(1, num_restarts)
      frac = cyc - math.floor(cyc)
      if progress >= 1.0:
        frac = 1.0  # final point lands on the trough, not a fresh peak
      return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(f"unknown lr_scheduler: {scheduler!r}")

  return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# --------------------------------------------------------------------------- #
# Held-out validation loss (deterministic, low-variance).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_val_loss(
  loss_fn,
  batches: list,
  *,
  quantiles: tuple = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> float:
  """Mean flow-matching loss over held-out ``batches`` at FIXED timestep quantiles.

  ``loss_fn(batch, q) -> scalar tensor`` runs a PLAIN (unmasked, unweighted) flow loss
  for ``batch`` at schedule quantile ``q`` in [0, 1], using a deterministic (fixed-seed)
  noise generator. Evaluating at the schedule's quantile timesteps rather than random t
  makes this a low-variance generalization signal (vs the noisy random-t training loss).

  ``batches`` is a list of ``(batch, n)`` pairs where ``n`` weights the per-batch loss
  in the sample-weighted mean (so an uneven final batch counts correctly). The caller
  owns batching, conditioning, and the schedule (closed over by ``loss_fn``); this stays
  model-agnostic so the same evaluator serves t2i and edit/multiref. Returns the mean,
  or NaN if there are no batches.
  """
  if not batches:
    return float("nan")
  total, count = 0.0, 0
  for q in quantiles:
    for batch, n in batches:
      total += float(loss_fn(batch, q)) * n
      count += n
  return total / max(1, count)


# --------------------------------------------------------------------------- #
# Optimizer factory (adds Prodigy = auto-LR).
# --------------------------------------------------------------------------- #
def build_optimizer(name: str, params, lr: float, *, weight_decay: float = 0.0):
  """Build an optimizer by name: adamw | adamw8bit | prodigy | schedule_free | came.

  prodigy / schedule_free adapt their own learning rate (lr ~1.0 / a normal lr).
  schedule_free manages its own schedule and needs opt.train()/opt.eval() around
  the training/eval phases (see is_schedule_free + the trainer wiring). Non-core
  backends are imported lazily.
  """
  name = name.lower()
  if name == "adamw":
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
  if name == "adamw8bit":
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
  if name == "prodigy":
    from prodigyopt import Prodigy
    # Prodigy: lr acts as a scaling on the adapted step; 1.0 is the canonical value.
    return Prodigy(params, lr=lr, weight_decay=weight_decay, safeguard_warmup=True)
  if name == "schedule_free":
    from schedulefree import AdamWScheduleFree
    return AdamWScheduleFree(params, lr=lr, weight_decay=weight_decay)
  if name == "came":
    from pytorch_optimizer import CAME
    return CAME(params, lr=lr, weight_decay=weight_decay)
  raise ValueError(
    f"unknown optimizer: {name!r} (adamw | adamw8bit | prodigy | schedule_free | came)"
  )


def is_schedule_free(name: str) -> bool:
  """schedule_free optimizers need opt.train()/opt.eval() and no LR scheduler."""
  return name.lower() == "schedule_free"


# --------------------------------------------------------------------------- #
# EMA of trainable weights (CPU-resident -> zero extra VRAM).
# --------------------------------------------------------------------------- #
class EmaModel:
  """Exponential moving average of a set of live tensors, kept in CPU RAM (fp32).

  Construct with ``named_tensors`` = ``{name: live Tensor}`` (the actual parameters,
  not copies -- the optimizer updates them in place and ``update`` reads them back).
  For a full-FT model pass the float entries of ``model.state_dict()``; for LoRA pass
  the adapter tensors. The shadow lives on the host, so a 12B EMA costs ~0 VRAM.

  ``every`` strides the host copy (the decay is compounded as ``decay**every`` so the
  effective time-constant is unchanged) -- a per-step copy of every weight would
  otherwise dominate step time on a large full-FT.

  ``store()/copy_to()/restore()`` swap the EMA weights into the live tensors for
  sampling or a format-specific save, then put the originals back. ``write_safetensors``
  serialises the EMA weights directly (no swap, no VRAM spike) given a key template.
  """

  def __init__(self, named_tensors, decay: float = 0.999, *, every: int = 1):
    self.decay = float(decay)
    self.every = max(1, int(every))
    self.live = dict(named_tensors)
    self.shadow = {k: v.detach().to("cpu", torch.float32).clone() for k, v in self.live.items()}
    self._backup = None

  @torch.no_grad()
  def update(self, step: int) -> None:
    """Fold the live weights into the shadow (only on stride boundaries)."""
    if (step + 1) % self.every != 0:
      return
    d = self.decay ** self.every
    for k, v in self.live.items():
      self.shadow[k].mul_(d).add_(v.detach().to("cpu", torch.float32), alpha=1.0 - d)

  def state_dict(self) -> dict:
    return {"decay": self.decay, "every": self.every, "shadow": self.shadow}

  def load_state_dict(self, sd: dict) -> None:
    self.decay = sd.get("decay", self.decay)
    self.every = sd.get("every", self.every)
    self.shadow = sd["shadow"]

  @torch.no_grad()
  def store(self) -> None:
    """Snapshot the live tensors so :meth:`restore` can undo a :meth:`copy_to`."""
    self._backup = {k: v.detach().clone() for k, v in self.live.items()}

  @torch.no_grad()
  def copy_to(self) -> None:
    """Overwrite the live tensors with the EMA weights (call :meth:`store` first)."""
    for k, v in self.live.items():
      v.detach().copy_(self.shadow[k].to(v.device, v.dtype))

  @torch.no_grad()
  def restore(self) -> None:
    if self._backup is None:
      return
    for k, v in self.live.items():
      v.detach().copy_(self._backup[k])
    self._backup = None

  @torch.no_grad()
  def write_safetensors(self, path, template: dict, *, dtype=torch.bfloat16, metadata=None) -> None:
    """Save EMA weights as a standalone safetensors. ``template`` = the full key set
    (e.g. ``model.state_dict()``); keys absent from the shadow (non-float buffers) fall
    back to their live value so the file is a complete, loadable checkpoint."""
    from safetensors.torch import save_file

    out = {}
    for k, v in template.items():
      src = self.shadow[k] if k in self.shadow else v.detach().to(torch.float32).cpu()
      out[k] = src.to(dtype).contiguous()
    tmp = f"{path}.tmp"
    save_file(out, tmp, metadata=metadata or {})
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Resumable training state (optimizer + scheduler + RNG + step [+ EMA]).
# Weights are checkpointed separately (safetensors for full-FT, adapter file for LoRA);
# this captures everything else needed to continue a run from where it stopped.
# --------------------------------------------------------------------------- #
def save_resume_state(path, *, step, optimizer, scheduler=None, gen=None, rng=None, ema=None):
  """Atomically save the non-weight state needed to resume at ``step``."""
  blob = {
    "step": step,
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler is not None else None,
    "ema": ema.state_dict() if ema is not None else None,
    "rng_python": rng.getstate() if rng is not None else None,
    "rng_torch": torch.get_rng_state(),
    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    "gen": gen.get_state() if gen is not None else None,
  }
  tmp = f"{path}.tmp"
  torch.save(blob, tmp)
  os.replace(tmp, path)


def load_resume_state(path, *, optimizer, scheduler=None, gen=None, rng=None, ema=None,
                      offload=False, map_location="cpu"):
  """Restore a :func:`save_resume_state` blob in place. Returns the saved ``step``.

  ``offload``: keep the optimizer's Adam moments in CPU RAM after restore. torch's
  ``Optimizer.load_state_dict`` casts state tensors to each param's device, which would
  pull an offloaded run's moments onto the GPU; set this to match the full-FT
  ``offload_optimizer`` backend so resume preserves the low-VRAM layout. The on-GPU
  Adafactor backend (the default) wants its state on the GPU, so leave it False there.
  RNG-restore failures never abort a resume (they only affect noise reproducibility).
  """
  blob = torch.load(path, map_location=map_location, weights_only=False)
  optimizer.load_state_dict(blob["optimizer"])
  if offload:
    for st in optimizer.state.values():
      for key in ("exp_avg", "exp_avg_sq"):
        t = st.get(key)
        if isinstance(t, torch.Tensor) and t.is_cuda:
          cpu_t = t.to("cpu")
          st[key] = cpu_t.pin_memory() if torch.cuda.is_available() else cpu_t
  if scheduler is not None and blob.get("scheduler") is not None:
    scheduler.load_state_dict(blob["scheduler"])
  if ema is not None and blob.get("ema") is not None:
    ema.load_state_dict(blob["ema"])
  if rng is not None and blob.get("rng_python") is not None:
    rng.setstate(blob["rng_python"])
  # RNG states must be CPU ByteTensors -- map_location may have moved them to CUDA.
  try:
    torch.set_rng_state(blob["rng_torch"].cpu().to(torch.uint8))
    if blob.get("rng_cuda") is not None and torch.cuda.is_available():
      torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in blob["rng_cuda"]])
    if gen is not None and blob.get("gen") is not None:
      gen.set_state(blob["gen"].cpu().to(torch.uint8))
  except Exception as e:
    print(f"[resume] RNG state restore skipped ({e}); continuing", flush=True)
  return blob["step"]


# --------------------------------------------------------------------------- #
# NaN / inf guard.
# --------------------------------------------------------------------------- #
def is_finite_loss(loss: torch.Tensor) -> bool:
  """True iff the loss is finite (no NaN/inf) -- skip the optimizer step if False."""
  return bool(torch.isfinite(loss).all())
