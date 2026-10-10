#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base) on the tennis demonstrations (Colab GPU).
#   STEPS=20000 BATCH=64 DATA=/content/data/tennis_vla OUT=/content/train bash scripts/vla/colab/train.sh
set -euo pipefail
STEPS=${STEPS:-20000}
BATCH=${BATCH:-64}
DATA=${DATA:-/content/data/tennis_vla}
OUT=${OUT:-/content/train}
export ACCELERATE_MIXED_PRECISION=${AMP:-bf16}
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/tennis_vla --dataset.root="$DATA" \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq=5000 --log_freq=100 --num_workers=10 \
  --output_dir="$OUT" --job_name=smolvla_tennis \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.front_prev": "observation.images.camera2"}'