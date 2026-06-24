"""Functional flow-matching sampler for the K2 MMDiT (no Scheduler class)."""

import math

import torch
from einops import rearrange, repeat
from PIL import Image


def roundup(value, multiple, name):
    """Round `value` up to the nearest multiple, logging when padding is applied."""
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(
            f"[sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}"
        )
    return aligned


def prepare(img, txtlen, patch, txtmask):
    """Patchify the latent and build the combined text+image position / mask tensors.

    Returns (img_tokens, pos, mask).
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0).

    `mu` is interpolated linearly in image-sequence length between (x1,y1) and
    (x2,y2), then used to time-shift a uniform 1->0 grid. Pass an explicit `mu`
    to pin a constant shift regardless of resolution (used by the distilled
    checkpoint, which was trained at a fixed mu=1.15).
    """
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


@torch.no_grad()
def sample(
    model,
    ae,
    encoder,
    prompts,
    *,
    negative_prompts=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    images=None,
):
    """End-to-end text-to-image sampling: encode -> euler+CFG denoise -> decode.

    ``images`` (style-reference): one reference image per prompt fed through the VLM encoder so
    image-derived conditioning enters the text stream; the unconditional CFG branch stays text-only.
    """
    patch = model.config.patch

    # The latent grid (dim // ae.compression) is patchified in `patch`-sized blocks,
    # so width/height must be multiples of ae.compression * patch. Pad up otherwise.
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    # Per-prompt seeded gaussian latent noise.
    noise = torch.cat(
        [
            torch.randn(
                1,
                ae.channels,
                height // ae.compression,
                width // ae.compression,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    # Positive (conditional) text conditioning (+ optional style-reference image).
    txt, txtmask = encoder(prompts, images=images) if images is not None else encoder(prompts)
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    # The unconditional branch is only used for CFG; skip encoding/prep entirely
    # when guidance is disabled.
    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    # min_res/max_res define the (x1,y1)-(x2,y2) interpolation endpoints for `mu`.
    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # Euler integration of the flow ODE with CFG.
    img = x
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
        if cfg:
            uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
            v = cond + guidance * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v

    # Unpatchify back to a latent and decode to pixels.
    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (ae.compression * patch),
        w=width // (ae.compression * patch),
    )
    img = ae.decode(img.to(torch.bfloat16))
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]


@torch.no_grad()
def sample_regions(
    model,
    ae,
    encoder,
    regions,
    *,
    device="cuda",
    dtype=torch.bfloat16,
    width=1280,
    height=768,
    steps=28,
    guidance=4.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    isolate_regions=True,
):
    """Attention-couple regional sampling: one coherent latent, per-region conditioning.

    ``regions`` is a list of ``{"prompt": str, "box": (x0, y0, x1, y1)}`` with box edges as 0..1
    fractions of width/height. The region prompts are concatenated into one text stream and the DiT
    attention is masked so each region's image tokens attend only to their own region prompt (and to
    all image tokens, keeping the canvas coherent), while the text segments stay mutually isolated.
    Two identities in two boxes therefore cannot blend. This drives the model through the opt-in
    ``attn_mask_override`` / ``txt_attn_override`` kwargs; the default sampler path is untouched.
    """
    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")
    comp = ae.compression
    h_lat, w_lat = height // comp, width // comp
    h_, w_ = h_lat // patch, w_lat // patch
    imglen = h_ * w_

    noise = torch.randn(
        1, ae.channels, h_lat, w_lat, device=device, dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    # Encode each region prompt; keep only real (unpadded) tokens, concat into one text stream.
    enc, emask = encoder([r["prompt"] for r in regions])      # (R, Lmax, 12, 2560), (R, Lmax)
    parts = [enc[i][emask[i]] for i in range(len(regions))]    # each (Lr, 12, 2560)
    seglens = [p.shape[0] for p in parts]
    context = torch.cat(parts, dim=0).unsqueeze(0)             # (1, txtlen, 12, 2560)
    txtlen = context.shape[1]
    L = txtlen + imglen

    # Region of each image token (row-major h,w order matches prepare()).
    yy = torch.arange(h_, device=device).view(h_, 1).expand(h_, w_).reshape(-1)
    xx = torch.arange(w_, device=device).view(1, w_).expand(h_, w_).reshape(-1)
    region_of = torch.full((imglen,), -1, dtype=torch.long, device=device)
    for ri, r in enumerate(regions):
        x0, y0, x1, y1b = r["box"]
        inbox = ((xx >= int(x0 * w_)) & (xx < max(int(x1 * w_), int(x0 * w_) + 1)) &
                 (yy >= int(y0 * h_)) & (yy < max(int(y1b * h_), int(y0 * h_) + 1)))
        region_of[inbox] = ri          # later regions win on overlap

    # (L, L) boolean attention-allow matrix. ``isolate_regions`` decides image<->image scope:
    # False = one coherent canvas (global image attention; tends to merge two same-class subjects
    # into one centered figure); True = each region's image tokens attend ONLY their own region
    # (two independent sub-images) so two single-subject prompts render as two separate subjects.
    allowed = torch.zeros(L, L, dtype=torch.bool, device=device)
    imgvec = torch.zeros(L, dtype=torch.bool, device=device); imgvec[txtlen:] = True
    if not isolate_regions:
        allowed |= imgvec[:, None] & imgvec[None, :]           # global image <-> image
    off = 0
    for ri, Lr in enumerate(seglens):
        seg = torch.zeros(L, dtype=torch.bool, device=device); seg[off:off + Lr] = True
        allowed |= seg[:, None] & seg[None, :]                 # text segment self-attention
        rrow = torch.zeros(L, dtype=torch.bool, device=device)
        rrow[txtlen:][region_of == ri] = True                  # this region's image tokens
        allowed |= rrow[:, None] & seg[None, :]                # image-region -> its own text
        allowed |= seg[:, None] & rrow[None, :]                # and back
        if isolate_regions:
            allowed |= rrow[:, None] & rrow[None, :]           # image-region attends only itself
        off += Lr
    unc = region_of < 0
    if unc.any():                                              # uncovered tokens see all text (+ all image when isolated)
        urow = torch.zeros(L, dtype=torch.bool, device=device); urow[txtlen:][unc] = True
        tcol = torch.zeros(L, dtype=torch.bool, device=device); tcol[:txtlen] = True
        allowed |= urow[:, None] & tcol[None, :]; allowed |= tcol[:, None] & urow[None, :]
        if isolate_regions:
            allowed |= urow[:, None] & imgvec[None, :]; allowed |= imgvec[:, None] & urow[None, :]

    attn_override = allowed.unsqueeze(0).unsqueeze(0)          # (1,1,L,L)
    txt_override = allowed[:txtlen, :txtlen].unsqueeze(0).unsqueeze(0)

    keypad = torch.ones(1, txtlen, dtype=torch.bool, device=device)
    x, pos, mask = prepare(noise, txtlen, patch, keypad)

    cfg = guidance > 0
    if cfg:
        untxt, untxtmask = encoder([""])
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1e = (minres // align) ** 2
    x2e = (maxres // align) ** 2
    ts = timesteps(x.shape[1], steps, x1e, x2e, y1=y1, y2=y2, mu=mu)

    img = x
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t = torch.full((1,), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=context, t=t, pos=pos, mask=mask,
                     attn_mask_override=attn_override, txt_attn_override=txt_override)
        if cfg:
            uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
            v = cond + guidance * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v

    img = rearrange(
        img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_
    )
    img = ae.decode(img.to(torch.bfloat16))
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]
