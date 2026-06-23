"""Krea 2 architecture, packing, and flow-matching constants.

Derived from the reference code (``mmdit.py``, ``encoder.py``, ``autoencoder.py``,
``sampling.py``) and ``krea/Krea-2-Raw`` ``model_index.json``. Single source of truth
imported by the trainer / precache / sampler.
"""

# --- Text encoder (Qwen3-VL-4B-Instruct), see encoder.py -------------------------------------
# Hidden-state layers tapped per token; stacked on a separate axis (NOT flattened) -> (B,L,12,2560).
QWEN3_VL_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)  # 12 layers
NUM_TEXT_LAYERS = len(QWEN3_VL_SELECT_LAYERS)  # 12  == SingleMMDiTConfig.txtlayers
QWEN3_VL_HIDDEN = 2560                          # == SingleMMDiTConfig.txtdim
TEXT_MAX_LENGTH = 512                           # encoder.py TextEncoderConfig.max_length
TEXT_ENCODER_ID = "Qwen/Qwen3-VL-4B-Instruct"

# --- VAE (AutoencoderKLQwenImage), see autoencoder.py ----------------------------------------
VAE_ID = "Qwen/Qwen-Image"        # subfolder="vae"
VAE_COMPRESSION = 8               # f8 spatial downsample
VAE_CHANNELS = 16                 # latent channels
PATCH_SIZE = 2                    # SingleStreamDiT patchify (single_mmdit_large_wide.patch)
PIXEL_PER_TOKEN = VAE_COMPRESSION * PATCH_SIZE          # 16: pixels per image token, per side
LATENT_TOKEN_DIM = VAE_CHANNELS * PATCH_SIZE * PATCH_SIZE  # 64: per-token latent dim fed to DiT.first

# --- DiT (SingleStreamDiT / single_mmdit_large_wide), see mmdit.py + inference.py ------------
DIT_FEATURES = 6144
DIT_LAYERS = 28
DIT_HEADS = 48
DIT_KVHEADS = 12                  # GQA
DIT_MULTIPLIER = 4
DIT_TDIM = 256

# --- Flow-matching dynamic timestep shift (Krea res-aware "mu" shift) -------------------------
# FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True) equivalent params from the diffusers
# Krea2 pipeline. mu is linearly interpolated by IMAGE-TOKEN COUNT (grid_h*grid_w), not pixels.
SHIFT_BASE = 0.5                  # mu at SHIFT_BASE_SEQ_LEN
SHIFT_MAX = 1.15                  # mu at SHIFT_MAX_SEQ_LEN
SHIFT_BASE_SEQ_LEN = 256
SHIFT_MAX_SEQ_LEN = 6400
SHIFT_SIGMA = 1.0

# --- Flow-matching convention (matches sampling.py) ------------------------------------------
#   t = 1 -> pure noise,  t = 0 -> data
#   x_t       = t * noise + (1 - t) * x0
#   v_target  = noise - x0          (the velocity the DiT predicts)
#   euler     : x <- x + v * (t_next - t_curr)   with t stepping 1 -> 0
