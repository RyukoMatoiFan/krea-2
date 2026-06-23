"""Text-to-image flow-matching training step for the Krea 2 ``SingleStreamDiT``.

The DiT forward is ``dit(img, context, t, pos, mask)`` where (see mmdit.py):
  * ``img``     : (B, n, 64)        patchified latent image tokens (the noised target x_t)
  * ``context`` : (B, L, 12, 2560)  the 12 tapped Qwen3-VL hidden states (text conditioning)
  * ``t``       : (B,)              flow time in [0, 1]
  * ``pos``     : (B, L+n, 3)       3-axis rotary coords (text at origin, image at grid (0,h,w))
  * ``mask``    : (B, L+n) bool     valid-token mask (real text tokens + all image tokens)
and it returns the predicted velocity over the IMAGE tokens only: (B, n, 64).

Flow convention (matches sampling.py / constants.py): t=1 noise, t=0 data,
``x_t = t*noise + (1-t)*z0``, ``v_target = noise - z0``.

The text conditioning is produced either live (joint trainer: the trainable Qwen3-VL encodes the
caption each step) or from a precomputed cache (DiT-only trainer: frozen TE). Both feed the same step.

Reference-extensible: :func:`build_pos_mask` accepts extra image grids so edit / multi-reference
training can pack source/reference latent tokens into the same stream. The plain t2i path passes a
single (grid_h, grid_w).
"""
from __future__ import annotations

import torch
from torch import Tensor

from training_utils import flow_loss, timestep_weight


def build_pos_mask(
    grid_h: int,
    grid_w: int,
    text_mask: Tensor,
    *,
    ref_grids: list[tuple[int, int]] | None = None,
) -> tuple[Tensor, Tensor]:
    """Build combined ``pos`` (B, S, 3) and ``mask`` (B, S) for ``[text][refs...][target]``.

    ``text_mask`` is (B, L) bool. Image tokens are all valid. Image position ids carry their
    latent-grid coords on axes (1, 2); axis 0 (temporal) is 0 for t2i, and is bumped per reference
    so reference grids never collide with the target grid (edit/multiref use this).
    """
    B, L = text_mask.shape
    device = text_mask.device
    grids = [(grid_h, grid_w)] if not ref_grids else [*ref_grids, (grid_h, grid_w)]

    pos_blocks = [torch.zeros(L, 3, device=device)]  # text at origin
    mask_blocks = [text_mask]
    for t_idx, (gh, gw) in enumerate(grids):
        ids = torch.zeros((gh, gw, 3), device=device)
        ids[..., 0] = t_idx  # temporal/stream index: 0 = target for plain t2i
        ids[..., 1] = torch.arange(gh, device=device)[:, None]
        ids[..., 2] = torch.arange(gw, device=device)[None, :]
        pos_blocks.append(ids.reshape(gh * gw, 3))
        mask_blocks.append(torch.ones(B, gh * gw, dtype=torch.bool, device=device))

    pos = torch.cat(pos_blocks, dim=0).unsqueeze(0).expand(B, -1, -1)  # (B, S, 3)
    mask = torch.cat(mask_blocks, dim=1)                              # (B, S)
    return pos, mask


def sample_timesteps(schedule, bsz: int, device, generator=None) -> Tensor:
    """Draw a shifted training timestep per sample: ``t = schedule(U(0,1))`` (Krea dynamic shift)."""
    u = torch.rand(bsz, device=device, generator=generator)
    return schedule(u).to(device)


def t2i_training_step(
    dit,
    *,
    z0: Tensor,            # (B, n, 64) patchified, normalized latent target
    context: Tensor,      # (B, L, 12, 2560) text conditioning
    text_mask: Tensor,    # (B, L) bool
    grid_h: int,
    grid_w: int,
    schedule,             # KreaShiftSchedule for this bucket's image-token count
    flow_cfg,             # FlowConfig (timestep_weighting, min_snr_gamma, noise_offset, input_perturbation)
    generator=None,
    cfg_dropout_prob: float = 0.0,
    t_override: float | None = None,
    disable_weighting: bool = False,
) -> Tensor:
    """One flow-matching loss over the image tokens. Returns a scalar loss (keeps the graph).

    ``disable_weighting`` skips the per-timestep loss weighting -> a plain MSE, used for the
    deterministic held-out validation signal so it is comparable across schemes/steps.
    """
    B, n, dim = z0.shape
    device = z0.device

    # Timestep (per sample), in float32 for the flow math.
    if t_override is not None:
        t = torch.full((B,), float(t_override), device=device, dtype=torch.float32)
    else:
        t = sample_timesteps(schedule, B, device, generator).float()

    # Noise (+ optional per-channel offset), build x_t and the velocity target in float32.
    noise = torch.randn(z0.shape, device=device, generator=generator, dtype=torch.float32)
    if flow_cfg.noise_offset:
        noise = noise + flow_cfg.noise_offset
    z0f = z0.float()
    tt = t.view(B, 1, 1)
    x_t = tt * noise + (1.0 - tt) * z0f
    if flow_cfg.input_perturbation:
        x_t = x_t + flow_cfg.input_perturbation * torch.randn_like(x_t)
    v_target = noise - z0f

    # Classifier-free guidance dropout: zero the text conditioning for dropped samples.
    ctx = context
    if cfg_dropout_prob > 0.0:
        drop = torch.rand(B, device=device, generator=generator) < cfg_dropout_prob
        if drop.any():
            ctx = context.clone()
            ctx[drop] = 0

    pos, mask = build_pos_mask(grid_h, grid_w, text_mask)

    pred = dit(
        img=x_t.to(dit_dtype(dit)),
        context=ctx.to(dit_dtype(dit)),
        t=t,
        pos=pos,
        mask=mask,
    )  # (B, n, 64)

    weight = None if disable_weighting else timestep_weight(
        t, flow_cfg.timestep_weighting, gamma=flow_cfg.min_snr_gamma)
    return flow_loss(pred.float(), v_target, weight=weight)


def edit_training_step(
    dit,
    *,
    z0: Tensor,                 # (B, n_tgt, 64) target latent
    refs: list,                 # list of (B, n_ref_i, 64) CLEAN reference latents (edit source / multiref)
    ref_grids: list,            # [(gh_i, gw_i)] for each reference, in the same order as `refs`
    context: Tensor,            # (B, L, 12, 2560)
    text_mask: Tensor,          # (B, L)
    grid_h: int,
    grid_w: int,
    schedule,
    flow_cfg,
    generator=None,
    cfg_dropout_prob: float = 0.0,
    ref_dropout_prob: float = 0.0,
    t_override: float | None = None,
    disable_weighting: bool = False,
    loss_mask: Tensor | None = None,
) -> Tensor:
    """Flow step for edit / multi-reference: pack ``[text, refs(clean), target(noised)]``, loss on target.

    References are inserted as CLEAN latent tokens with their own rotary position blocks (via
    ``ref_grids``); only the target is noised at ``t`` and supervised. No model change -- the DiT
    outputs velocity for all image tokens and we slice the trailing target block. The model learns to
    treat reference tokens as clean conditioning (in-context editing). ``ref_dropout_prob`` zeros the
    reference tokens per-sample so a no-reference (text-only) branch is also learned for CFG.
    """
    B, n_tgt, _ = z0.shape
    device = z0.device
    if t_override is not None:
        t = torch.full((B,), float(t_override), device=device, dtype=torch.float32)
    else:
        t = sample_timesteps(schedule, B, device, generator).float()

    noise = torch.randn(z0.shape, device=device, generator=generator, dtype=torch.float32)
    if flow_cfg.noise_offset:
        noise = noise + flow_cfg.noise_offset
    z0f = z0.float()
    tt = t.view(B, 1, 1)
    x_t = tt * noise + (1.0 - tt) * z0f
    if flow_cfg.input_perturbation:
        x_t = x_t + flow_cfg.input_perturbation * torch.randn_like(x_t)
    v_target = noise - z0f

    ctx = context
    if cfg_dropout_prob > 0.0:
        drop = torch.rand(B, device=device, generator=generator) < cfg_dropout_prob
        if drop.any():
            ctx = context.clone()
            ctx[drop] = 0

    ref_tokens = [r.float() for r in refs]
    if ref_dropout_prob > 0.0 and ref_tokens:
        rdrop = torch.rand(B, device=device, generator=generator) < ref_dropout_prob
        if rdrop.any():
            ref_tokens = [r.clone() for r in ref_tokens]
            for r in ref_tokens:
                r[rdrop] = 0

    img = torch.cat([*ref_tokens, x_t], dim=1)  # [refs..., target] along the sequence
    pos, mask = build_pos_mask(grid_h, grid_w, text_mask, ref_grids=ref_grids)

    out = dit(img=img.to(dit_dtype(dit)), context=ctx.to(dit_dtype(dit)), t=t, pos=pos, mask=mask)
    pred_tgt = out[:, -n_tgt:].float()  # supervise only the trailing target tokens
    weight = None if disable_weighting else timestep_weight(
        t, flow_cfg.timestep_weighting, gamma=flow_cfg.min_snr_gamma)
    # loss_mask (edit masked loss): per-token (B, n_tgt) weights concentrating loss on the edited region.
    return flow_loss(pred_tgt, v_target, mask=loss_mask, weight=weight)


def dit_dtype(dit) -> torch.dtype:
    """Compute dtype of the DiT (first parameter's dtype)."""
    return next(dit.parameters()).dtype
