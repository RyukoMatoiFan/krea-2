#!/usr/bin/env python
"""Edit / multi-reference / style sampling for Krea 2: clean reference(s) -> denoise the target.

Mirrors ``edit_training_step``: the reference image(s) are VAE-encoded to CLEAN latent tokens and
packed ahead of the noised target in the single stream (each at its own RoPE frame via
``build_pos_mask``); only the trailing target tokens are denoised. Use it for instruction edits
(ref = source image), multi-reference composition, and style transfer (ref = style exemplar).

  python sample_edit.py --config config/t2i_lora.yaml --lora runs/r/ckpts/lora_final.safetensors \
      --prompt "make it autumn" --ref source.jpg --out edited.png
  python sample_edit.py --config ... --prompt "a cat on a beach" --ref style.jpg --out styled.png
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image
from safetensors.torch import load_file

from loading import build_dit, build_encoder, build_vae
from lora import load_lora
from precache_t2i import encode_latent, images_to_tensor, patchify
from sampling import roundup, timesteps
from train_t2i import build_pos_mask
from training_config import apply_runtime, dtype_of, load_config


@torch.no_grad()
def edit_sample(model, vae, encoder, prompt, ref_images, *, negative_prompt="", device="cuda",
                dtype=torch.bfloat16, width=1024, height=1024, steps=28, guidance=4.5, seed=0,
                minres=256, maxres=1280, y1=0.5, y2=1.15, mu=None):
    """Denoise a single target conditioned on a text prompt + clean reference image(s)."""
    patch = model.config.patch
    align = vae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")
    glat_h, glat_w = height // align, width // align  # target latent grid (in patches)

    # Clean reference tokens (square-resized to the target size, matching precache_edit).
    ref_tokens, ref_grids = [], []
    for im in ref_images:
        x = images_to_tensor(im, width, height, dtype).to(device)  # (1,3,H,W) in [-1,1]
        z = encode_latent(vae, x)                                   # (1,16,h,w) normalized
        ref_tokens.append(patchify(z, patch).to(dtype))             # (1, n_ref, 64)
        ref_grids.append((glat_h, glat_w))

    # Target noise (patchified latent), seeded.
    noise = torch.randn(1, vae.channels, height // vae.compression, width // vae.compression,
                        device=device, dtype=dtype,
                        generator=torch.Generator(device=device).manual_seed(seed))
    x_t = patchify(noise, patch)            # (1, n_tgt, 64)
    n_tgt = x_t.shape[1]

    cfg = guidance > 0
    txt, txtmask = encoder([prompt])
    pos, mask = build_pos_mask(glat_h, glat_w, txtmask, ref_grids=ref_grids)
    if cfg:
        untxt, untxtmask = encoder([negative_prompt])
        unpos, unmask = build_pos_mask(glat_h, glat_w, untxtmask, ref_grids=ref_grids)

    # mu-shift uses the TARGET image-token count (matches get_schedule_for_seqlen in training).
    x1 = (minres // align) ** 2
    x2 = (maxres // align) ** 2
    ts = timesteps(n_tgt, steps, x1, x2, y1=y1, y2=y2, mu=mu)

    def velocity(context, p, m):
        img = torch.cat([*ref_tokens, x_t], dim=1)   # [refs(clean), target(noised)]
        out = model(img=img.to(dtype), context=context.to(dtype), t=t, pos=p, mask=m)
        return out[:, -n_tgt:].float()                # supervise only the trailing target

    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        t = torch.full((1,), tcurr, dtype=dtype, device=device)
        v = velocity(txt, pos, mask)
        if cfg:
            v = v + guidance * (v - velocity(untxt, unpos, unmask))
        x_t = (x_t.float() + (tprev - tcurr) * v).to(dtype)

    from einops import rearrange
    lat = rearrange(x_t, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch,
                    h=glat_h, w=glat_w)
    img = vae.decode(lat.to(torch.bfloat16)).clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return Image.fromarray(img[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ref", action="append", required=True, help="reference image(s); repeat for multiref")
    ap.add_argument("--negative", default="")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5, help="CFG scale (0 disables CFG; use 0 for Turbo)")
    ap.add_argument("--mu", type=float, default=None, help="pin resolution-shift mu (Turbo: 1.15)")
    ap.add_argument("--y1", type=float, default=0.5, help="mu at min resolution")
    ap.add_argument("--y2", type=float, default=1.15, help="mu at max resolution")
    ap.add_argument("--base", default=None,
                    help="override base DiT repo/dir (e.g. krea/Krea-2-Turbo) -> run a Raw-trained LoRA on Turbo")
    ap.add_argument("--base-file", dest="base_file", default=None, help="weight filename within --base")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="edit.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)
    if args.base:           # load a different base (e.g. Turbo) before applying a Raw-trained LoRA
        cfg.paths.dit_repo = args.base
    if args.base_file:
        cfg.paths.dit_file = args.base_file

    dit = build_dit(cfg, device, dtype, load_weights=(args.ckpt is None), train=False)
    if args.ckpt:
        dit.load_state_dict(load_file(args.ckpt), strict=True, assign=False)
        dit = dit.to(device=device, dtype=dtype).eval()
    if args.lora:
        load_lora(dit, args.lora)
        dit = dit.to(device=device, dtype=dtype).eval()
    encoder = build_encoder(cfg, device, dtype, train=False)
    vae = build_vae(cfg, device, dtype)

    refs = [Image.open(p) for p in args.ref]
    img = edit_sample(dit, vae, encoder, args.prompt, refs, negative_prompt=args.negative,
                      device=device, dtype=dtype, width=args.width, height=args.height,
                      steps=args.steps, guidance=args.guidance, seed=args.seed,
                      mu=args.mu, y1=args.y1, y2=args.y2)
    img.save(args.out)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
