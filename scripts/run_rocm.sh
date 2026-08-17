#!/usr/bin/env bash
# Launch the full fine-tune on a ROCm host. Environment only -- no Python file reads anything set
# here, so this script cannot affect the CUDA path.
#
# Deliberately NOT a second config YAML. Every value below is expressible as a KREA2_<SECTION>__<KEY>
# override, and a parallel recipe file is how two recipes silently drift apart.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

: "${KREA2_TASK_DIR:?set KREA2_TASK_DIR to the task directory}"
: "${KREA2_CONFIG:?set KREA2_CONFIG to the training recipe YAML}"
CACHE="$KREA2_TASK_DIR/cache"

# --- allocator -------------------------------------------------------------------------------
# expandable_segments is deliberately NOT set here. It earns its keep only when a configuration
# runs close to its memory ceiling and fragmentation decides whether a step fits; with headroom to
# spare that pressure is gone, and it remains the less-travelled allocator path -- exactly where a
# silent allocation bug would live. Both names are exported so the value is unambiguous whichever
# one the build reads.
export PYTORCH_CUDA_ALLOC_CONF=""
export PYTORCH_HIP_ALLOC_CONF=""

# --- compile / kernel caches ----------------------------------------------------------------
# Persist across runs: a cold Inductor or Triton cache costs minutes of recompilation per launch,
# and on a rented pod that is billed time spent recompiling something already compiled.
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export TRITON_CACHE_DIR="$CACHE/triton"
# MIOpen tunes convolutions on first sight and caches the result; only the VAE (precache, previews)
# has convolutions, but a cold database makes the first preview look pathological.
export MIOPEN_USER_DB_PATH="$CACHE/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE/miopen"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH"

# --- recipe override --------------------------------------------------------------------------
# Activation checkpointing trades compute for memory, so with memory to spare it is pure cost:
# recomputing every block's forward is a substantial share of the step. Raise this if a step does
# not fit.
export KREA2_OPTIM__GRAD_CHECKPOINTING_BLOCKS=0

echo "== environment"
python - <<'PY'
import torch
print(f"   torch {torch.__version__}  hip={torch.version.hip}  cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"   device {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}")
PY

echo "== verification ladder (stops before wasting a rental on a broken stack)"
python rocm_verify.py

echo "== training"
exec python train_t2i_full_joint_cached.py --config "$KREA2_CONFIG" "$@"
