#!/usr/bin/env python
"""Train a Concept-Slider LoRA for Krea 2: a bidirectional attribute knob in velocity space.

No paired dataset — the slider regresses its ±adapter onto the FROZEN base's own velocity, nudged
along (positive − negative) at a neutral anchor::

    dir      = v_base(c+) − v_base(c−)            # frozen base, no grad
    target_± = v_base(anchor) ± eta·dir
    loss     = MSE(v_lora(+s), target_+) + MSE(v_lora(−s), target_−)

Context latents (the realistic x_t the slider probes) are self-generated rollouts (default) or a
folder of images (slider.rollouts=0 -> paths.data_root). At inference, dial the knob with
``lora_scaled(transformer, factor)`` (sample.py --lora-scale) — +factor enhances, −factor inverts.

  CUDA_VISIBLE_DEVICES=0 python train_slider.py --config config/slider.yaml
  # axis/strength via env, e.g. KREA2_SLIDER__POSITIVE='dark, dim' KREA2_SLIDER__NEGATIVE='bright'
"""
import argparse
import glob
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from constants import PATCH_SIZE
from loading import build_dit, build_encoder, build_vae
from lora import inject_lora, lora_parameters, lora_scaled, lora_disabled, save_lora
from precache_t2i import encode_latent, images_to_tensor, patchify
from sampling import sample
from training_config import apply_runtime, dtype_of, load_config

ROLLOUT_PROMPTS = [
    "a photograph of a living room with a sofa and a window",
    "a landscape with a house, trees and a path",
    "a portrait photograph of a person, plain background",
    "a street with parked cars and buildings",
    "a kitchen counter with fruit and utensils",
    "a desk with a laptop, books and a lamp",
    "a close-up of a flower with dew on the petals",
    "a wooden table with a bowl of fruit",
]
EVAL_PROMPTS = [
    "a photograph of a living room",
    "a landscape with a small house and trees",
    "a close-up portrait of a person's face",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/slider.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_runtime(cfg)
    device, dtype = cfg.runtime.device, dtype_of(cfg)
    s = cfg.slider
    res = int(cfg.data.resolution)
    rank = int(cfg.lora.rank)
    steps = int(cfg.optim.steps)
    out_dir, ckpt_dir = cfg.paths.output_dir, cfg.paths.ckpt_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    if not s.positive or not s.negative:
        raise SystemExit("slider.positive and slider.negative are required (the +/- attribute prompts)")

    dit = build_dit(cfg, device, dtype, load_weights=True, train=False)
    encoder = build_encoder(cfg, device, dtype, train=False)
    vae = build_vae(cfg, device, dtype)

    # 1) context latents
    z_ctxs = []
    if int(s.rollouts) > 0:
        print(f"[slider] generating {s.rollouts} rollout context images @ {res}", flush=True)
        for k in range(int(s.rollouts)):
            img = sample(dit, vae, encoder, [ROLLOUT_PROMPTS[k % len(ROLLOUT_PROMPTS)]],
                         width=res, height=res, steps=24, guidance=4.0, seed=1000 + k)[0]
            ten = images_to_tensor(img, res, res, dtype).to(device)
            z_ctxs.append(patchify(encode_latent(vae, ten), PATCH_SIZE)[0].to(torch.float32).cpu())
    else:
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths += glob.glob(os.path.join(cfg.paths.data_root, ext))
        if not paths:
            raise SystemExit(f"no context images under {cfg.paths.data_root} (set slider.rollouts>0)")
        for p in sorted(paths):
            ten = images_to_tensor(Image.open(p).convert("RGB"), res, res, dtype).to(device)
            z_ctxs.append(patchify(encode_latent(vae, ten), PATCH_SIZE)[0].to(torch.float32).cpu())
    n = z_ctxs[0].shape[0]
    gh = gw = res // (vae.compression * PATCH_SIZE)
    latent_dim = z_ctxs[0].shape[-1]
    print(f"[slider] {len(z_ctxs)} contexts, grid {gh}x{gw}={n}, dim {latent_dim}", flush=True)

    # 2) text features (frozen)
    def enc(p):
        t, m = encoder([p])
        return t.detach(), m.detach()
    txt_pos, m_pos = enc(s.positive)
    txt_neg, m_neg = enc(s.negative)
    txt_anc, m_anc = enc(s.anchor)
    print(f"[slider] (+)'{s.positive[:34]}' (-)'{s.negative[:34]}' anchor='{s.anchor or '<uncond>'}' "
          f"eta={s.eta} late_frac={s.late_frac}", flush=True)

    imgpos = torch.zeros(1, n, 3, device=device)
    imgpos[0, :, 1] = torch.arange(gh, device=device).view(gh, 1).expand(gh, gw).reshape(-1)
    imgpos[0, :, 2] = torch.arange(gw, device=device).view(1, gw).expand(gh, gw).reshape(-1)
    imgmask = torch.ones(1, n, device=device, dtype=torch.bool)

    def vel(x_t, txt, txtmask, t_scalar):
        L = txt.shape[1]
        pos = torch.cat([torch.zeros(1, L, 3, device=device), imgpos], dim=1)
        mask = torch.cat([txtmask, imgmask], dim=1)
        t_vec = torch.full((1,), float(t_scalar), device=device, dtype=x_t.dtype)
        return dit(img=x_t, context=txt, t=t_vec, pos=pos, mask=mask).float()

    # 3) train
    adapters = inject_lora(dit, rank=rank)
    params = lora_parameters(adapters)
    opt = torch.optim.AdamW(params, lr=float(cfg.optim.lr))
    print(f"[slider] {len(adapters)} adapters, {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)
    gen = torch.Generator(device=device).manual_seed(int(cfg.runtime.seed))
    rng = torch.Generator(device="cpu").manual_seed(int(cfg.runtime.seed))
    dit.gradient_checkpointing = True   # the bidirectional ±s grad passes are memory-heavy; ckpt fits 768+/1024
    dit.train()
    log_every = int(cfg.logging.log_every)
    for step in range(steps):
        z0 = z_ctxs[int(torch.randint(len(z_ctxs), (1,), generator=rng))].to(device).unsqueeze(0).to(dtype)
        noise = torch.randn(1, n, latent_dim, device=device, dtype=dtype, generator=gen)
        t = torch.rand((1,), device=device, generator=gen).item() * float(s.late_frac)
        x_t = t * noise + (1.0 - t) * z0
        with torch.no_grad(), lora_disabled(dit):
            v_anchor = vel(x_t, txt_anc, m_anc, t)
            direction = vel(x_t, txt_pos, m_pos, t) - vel(x_t, txt_neg, m_neg, t)
        target_plus = v_anchor + float(s.eta) * direction
        with lora_scaled(dit, float(s.train_scale)):
            v_plus = vel(x_t, txt_anc, m_anc, t)
        loss = F.mse_loss(v_plus, target_plus)
        if bool(s.bidirectional):
            target_minus = v_anchor - float(s.eta) * direction
            with lora_scaled(dit, -float(s.train_scale)):
                v_minus = vel(x_t, txt_anc, m_anc, t)
            loss = loss + F.mse_loss(v_minus, target_minus)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(cfg.optim.grad_clip))
        opt.step()
        if (step + 1) % log_every == 0:
            print(f"[slider] step {step+1}/{steps} loss {loss.item():.4f}", flush=True)

    name = (cfg.logging.run_name or "slider").replace("/", "_")
    out_lora = os.path.join(ckpt_dir, f"{name}_rank{rank}.safetensors")
    save_lora(adapters, out_lora, rank=rank, alpha=float(rank),
              metadata={"positive": s.positive, "negative": s.negative,
                        "eta": s.eta, "infer_scale": s.infer_scale, "type": "krea2-slider"})
    print(f"[slider] saved {out_lora}", flush=True)

    # 4) multi-scale [-s|off|+s] sweep (scene should hold; attribute should slide monotonically)
    scales = [float(x) for x in str(s.eval_scales).split(",") if x.strip()]
    print(f"[slider] rendering sweep at scales {scales}", flush=True)
    dit.eval()
    rows = []  # (label, [imgs])
    for sc in scales:
        for pi, prompt in enumerate(EVAL_PROMPTS):
            panels = []
            for factor in (-sc, 0.0, sc):
                with lora_scaled(dit, factor):
                    panels.append(sample(dit, vae, encoder, [prompt], width=res, height=res,
                                         steps=28, guidance=4.5, seed=4242 + pi)[0])
            rows.append((f"s={sc}  {prompt[:40]}", panels))
    w, h = rows[0][1][0].size
    strip = 22
    sheet = Image.new("RGB", (w * 3, (h + strip) * len(rows)), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for r, (label, panels) in enumerate(rows):
        y = r * (h + strip)
        d.text((4, y + 4), f"[-s | off | +s]  {label}", fill=(0, 0, 0))
        for c, im in enumerate(panels):
            sheet.paste(im, (c * w, y + strip))
    sweep = os.path.join(out_dir, f"{name}_sweep.png")
    sheet.save(sweep)
    print(f"[slider] saved {sweep}", flush=True)
    print("SLIDER_DONE", flush=True)


if __name__ == "__main__":
    main()
