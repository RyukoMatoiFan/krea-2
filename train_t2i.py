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

import re as _re

import torch
from torch import Tensor

from training_utils import flow_loss, timestep_weight


def build_pos_mask(
    grid_h: int,
    grid_w: int,
    text_mask: Tensor,
    *,
    ref_grids: list[tuple[int, int]] | None = None,
    ref_drop_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build combined ``pos`` (B, S, 3) and ``mask`` (B, S) for ``[text][refs...][target]``.

    ``text_mask`` is (B, L) bool. Image tokens are all valid. Image position ids carry their
    latent-grid coords on axes (1, 2); axis 0 (temporal) is 0 for the TARGET (same frame the base
    t2i model generates at, preserving its positional prior) and i+1 for reference i, so reference
    grids never collide with the target grid (edit/multiref use this — Kontext-style indexing).
    """
    B, L = text_mask.shape
    device = text_mask.device
    grids = [(grid_h, grid_w)] if not ref_grids else [*ref_grids, (grid_h, grid_w)]

    pos_blocks = [torch.zeros(L, 3, device=device)]  # text at origin
    mask_blocks = [text_mask]
    n_ref = len(grids) - 1  # the target grid is always last
    for j, (gh, gw) in enumerate(grids):
        ids = torch.zeros((gh, gw, 3), device=device)
        ids[..., 0] = 0 if j == n_ref else j + 1  # temporal/stream index: target -> 0, ref i -> i+1
        ids[..., 1] = torch.arange(gh, device=device)[:, None]
        ids[..., 2] = torch.arange(gw, device=device)[None, :]
        pos_blocks.append(ids.reshape(gh * gw, 3))
        # Reference positions are attention-masked for reference-dropped samples so the no-reference
        # branch is genuinely reference-free. Zeroing the latents is not enough: a zero token still
        # passes the biased input projection to a constant token occupying its own RoPE frame, from
        # which the model could read the reference count, geometry and frame identity.
        blk = torch.ones(B, gh * gw, dtype=torch.bool, device=device)
        if ref_drop_mask is not None and j != n_ref:      # target block (j == n_ref) always visible
            blk = blk & ~ref_drop_mask.to(device=device, dtype=torch.bool).view(B, 1)
        mask_blocks.append(blk)

    pos = torch.cat(pos_blocks, dim=0).unsqueeze(0).expand(B, -1, -1)  # (B, S, 3)
    mask = torch.cat(mask_blocks, dim=1)                              # (B, S)
    return pos, mask


def encode_text_with_ref_dropout(encoder, caps, vlm_imgs, drop_mask):
    """Encode captions with per-sample VLM source-image conditioning, honoring reference dropout.

    ``drop_mask`` (B,) bool: True samples are encoded WITHOUT the source image, so the no-reference
    branch carries no source signal through the text-encoder vision tokens. Pass the same mask to
    ``edit_training_step`` as ``ref_drop_mask`` so the latent refs are zeroed for the SAME samples —
    otherwise a "no-ref" sample would still see the source through the text conditioning. The two
    sub-batches (with / without image) are encoded separately and right-pad-merged back into the
    original order. Returns ``(ctx (B, L, 12, d), mask (B, L) bool)``.
    """
    keep = [i for i in range(len(caps)) if not bool(drop_mask[i])]
    drop = [i for i in range(len(caps)) if bool(drop_mask[i])]
    outs: dict = {}
    if keep:
        ctx_k, m_k = encoder([caps[i] for i in keep], images=[vlm_imgs[i] for i in keep])
        for j, i in enumerate(keep):
            outs[i] = (ctx_k[j], m_k[j])
    if drop:
        ctx_d, m_d = encoder([caps[i] for i in drop])
        for j, i in enumerate(drop):
            outs[i] = (ctx_d[j], m_d[j])
    L = max(v[0].shape[0] for v in outs.values())
    c0 = outs[next(iter(outs))][0]
    ctx = torch.zeros(len(caps), L, *c0.shape[1:], dtype=c0.dtype, device=c0.device)
    mask = torch.zeros(len(caps), L, dtype=torch.bool, device=c0.device)
    for i, (c, m) in outs.items():
        ctx[i, : c.shape[0]] = c
        mask[i, : m.shape[0]] = m.to(torch.bool)
    return ctx, mask


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



_ORD_RE = _re.compile(r"image[_ ]?(\d+)", _re.I)


def permute_reference_order(caption: str, n_refs: int, perm: list) -> str:
    """Rewrite ``image_k`` mentions to follow a reference permutation.

    ``perm`` is read as "new position i now holds the reference that was at ``perm[i]``". A caption
    that said ``image_{old+1}`` must therefore say ``image_{new+1}`` where ``perm[new] == old``.

    Why this exists: each reference occupies its own RoPE frame, so a model can satisfy the training
    objective by learning the DATASET's slot convention ("the background is always image_1") instead
    of reading the instruction. Permuting the references without rewriting the caption would teach
    the opposite of role binding; rewriting both together is what makes the ordinal informative.
    Rewriting is ATOMIC (single regex pass) -- sequential replacement collapses ordinals onto each
    other, e.g. 1->2 followed by 2->3 turns every original 1 into a 3.
    """
    inv = {old: new for new, old in enumerate(perm)}

    def sub(m):
        k = int(m.group(1)) - 1
        return f"image_{inv.get(k, k) + 1}" if 0 <= k < n_refs else m.group(0)

    return _ORD_RE.sub(sub, caption or "")


def augment_reference_order(
    samples: list[dict], rng, probability: float, *, enabled: bool
) -> list[dict]:
    """Apply train-only reference-order augmentation without mutating cached payloads.

    Validation passes ``enabled=False``. That path is a strict no-op and does not consume ``rng``,
    keeping fixed validation independent of the training sampler and evaluation cadence.
    """
    if not enabled or probability <= 0.0 or not samples or not samples[0].get("refs"):
        return samples
    out = samples
    for i, sample in enumerate(samples):
        n_refs = len(sample["refs"])
        if n_refs < 2 or rng.random() >= probability:
            continue
        perm = list(range(n_refs))
        rng.shuffle(perm)
        if perm == list(range(n_refs)):
            continue
        if out is samples:
            out = list(samples)
        changed = dict(sample)
        changed["refs"] = [sample["refs"][j] for j in perm]
        if sample.get("ref_paths"):
            changed["ref_paths"] = [sample["ref_paths"][j] for j in perm]
        changed["caption"] = permute_reference_order(sample.get("caption", ""), n_refs, perm)
        out[i] = changed
    return out


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
    ref_t0: bool = True,
    ref_drop_mask: Tensor | None = None,
) -> Tensor:
    """Flow step for edit / multi-reference: pack ``[text, refs(clean), target(noised)]``, loss on target.

    References are inserted as CLEAN latent tokens with their own rotary position blocks (via
    ``ref_grids``); only the target is noised at ``t`` and supervised. The DiT outputs velocity for
    all image tokens and we slice the trailing target block. ``ref_t0`` (default) tags the clean
    reference tokens with a t=0 timestep modulation inside the DiT so it reads them as clean
    conditioning (Kontext-style in-context edit); set False to modulate the reference tokens with the
    sampled ``t`` instead. ``ref_dropout_prob`` zeros the reference tokens per-sample so a no-reference (text-only)
    branch is also learned for CFG. ``ref_drop_mask`` (B,) bool overrides the internal draw — used when
    the caller already decided the dropout (VLM dual conditioning: the text encoding for dropped samples
    must also exclude the source image, see ``encode_text_with_ref_dropout``).
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
    rdrop = None
    if ref_drop_mask is not None:
        rdrop = ref_drop_mask.to(device=device, dtype=torch.bool)
    elif ref_dropout_prob > 0.0 and ref_tokens:
        rdrop = torch.rand(B, device=device, generator=generator) < ref_dropout_prob
    if rdrop is not None and ref_tokens and rdrop.any():
        ref_tokens = [r.clone() for r in ref_tokens]
        for r in ref_tokens:
            r[rdrop] = 0

    img = torch.cat([*ref_tokens, x_t], dim=1)  # [refs..., target] along the sequence
    # Pass the dropout mask so dropped references are ATTENTION-MASKED, not merely zeroed.
    pos, mask = build_pos_mask(grid_h, grid_w, text_mask, ref_grids=ref_grids,
                               ref_drop_mask=rdrop)

    # Number of leading (clean) reference image tokens -> the DiT gives them t=0 modulation when ref_t0.
    ref_len = sum(int(r.shape[1]) for r in ref_tokens) if ref_t0 else 0
    out = dit(img=img.to(dit_dtype(dit)), context=ctx.to(dit_dtype(dit)), t=t, pos=pos, mask=mask,
              ref_len=ref_len)
    pred_tgt = out[:, -n_tgt:].float()  # supervise only the trailing target tokens
    weight = None if disable_weighting else timestep_weight(
        t, flow_cfg.timestep_weighting, gamma=flow_cfg.min_snr_gamma)
    # loss_mask (edit masked loss): per-token (B, n_tgt) weights concentrating loss on the edited region.
    return flow_loss(pred_tgt, v_target, mask=loss_mask, weight=weight)


def dit_dtype(dit) -> torch.dtype:
    """Compute dtype of the DiT (first parameter's dtype)."""
    return next(dit.parameters()).dtype
