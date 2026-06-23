"""Build the Krea 2 models (DiT + text encoder + VAE) from a :class:`TrainConfig`.

The DiT is the forked ``SingleStreamDiT`` (the ``single_mmdit_large_wide`` config from the reference
``inference.py``); its weights load from the single ``raw.safetensors`` of ``krea/Krea-2-Raw``,
resolved via ``huggingface_hub`` (served from the local HF cache when present). The text encoder is the
frozen Qwen3-VL-4B conditioner (``encoder.py``); the VAE is ``AutoencoderKLQwenImage`` (``autoencoder.py``).

Krea 2 Raw is clean bf16, so there is no fp8 dequant: full fine-tune just loads the
state dict and flips ``requires_grad_``.
"""
from __future__ import annotations

import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from autoencoder import QwenAutoencoder
from constants import (
    DIT_FEATURES,
    DIT_HEADS,
    DIT_KVHEADS,
    DIT_LAYERS,
    DIT_MULTIPLIER,
    DIT_TDIM,
    NUM_TEXT_LAYERS,
    PATCH_SIZE,
    QWEN3_VL_HIDDEN,
    QWEN3_VL_SELECT_LAYERS,
    TEXT_MAX_LENGTH,
    VAE_CHANNELS,
)
from encoder import Qwen3VLConditioner
from mmdit import SingleMMDiTConfig, SingleStreamDiT


def krea2_dit_config() -> SingleMMDiTConfig:
    """The ``single_mmdit_large_wide`` config (from the reference inference.py)."""
    return SingleMMDiTConfig(
        features=DIT_FEATURES,
        tdim=DIT_TDIM,
        txtdim=QWEN3_VL_HIDDEN,
        heads=DIT_HEADS,
        kvheads=DIT_KVHEADS,
        multiplier=DIT_MULTIPLIER,
        layers=DIT_LAYERS,
        patch=PATCH_SIZE,
        channels=VAE_CHANNELS,
        txtheads=20,
        txtkvheads=20,
        txtlayers=NUM_TEXT_LAYERS,
    )


def resolve_dit_weights(dit_repo: str, dit_file: str) -> str:
    """Resolve the DiT weights to a local path (local dir/file or HF repo via the local cache)."""
    local = os.path.join(dit_repo, dit_file)
    if os.path.isfile(local):
        return local
    if os.path.isfile(dit_repo):
        return dit_repo
    return hf_hub_download(dit_repo, dit_file)


def build_dit(cfg, device, dtype, *, load_weights: bool = True, train: bool = False) -> SingleStreamDiT:
    """Construct ``SingleStreamDiT`` on meta and load ``raw.safetensors`` (assign=True, like inference.py)."""
    with torch.device("meta"):
        dit = SingleStreamDiT(krea2_dit_config())
    if load_weights:
        path = resolve_dit_weights(cfg.paths.dit_repo, cfg.paths.dit_file)
        dit.load_state_dict(load_file(path), strict=True, assign=True)
        dit = dit.to(device=device, dtype=dtype)
    else:
        dit = dit.to_empty(device=device).to(dtype=dtype)
    if train:
        dit.requires_grad_(True)
        dit.train()
    else:
        dit.eval().requires_grad_(False)
    return dit


def build_encoder(cfg, device, dtype, *, train: bool = False) -> Qwen3VLConditioner:
    """Build the frozen-by-default Qwen3-VL-4B conditioner (12-layer tap)."""
    enc = Qwen3VLConditioner(
        cfg.paths.text_encoder_id,
        max_length=TEXT_MAX_LENGTH,
        select_layers=QWEN3_VL_SELECT_LAYERS,
    )
    enc = enc.to(device=device, dtype=dtype)
    if train:
        # Joint full-FT: the underlying Qwen3-VL becomes trainable (see the joint trainer / te_lr).
        enc.qwen.requires_grad_(True)
        enc.qwen.train()
    else:
        enc.qwen.eval().requires_grad_(False)
    return enc


def build_vae(cfg, device, dtype) -> QwenAutoencoder:
    """Build the frozen AutoencoderKLQwenImage wrapper (encode for precache, decode for previews)."""
    vae = QwenAutoencoder()
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)
