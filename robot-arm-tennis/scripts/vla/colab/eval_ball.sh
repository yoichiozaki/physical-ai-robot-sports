#!/usr/bin/env bash
# Ablation: VLA that also receives the exact ball position/velocity (no need to see the ball).
# One process per (world, latency) so that the GPU is shared by many short inferences.
#   LATENCIES="0.0 0.1 0.2 0.36" bash scripts/vla/colab/eval_ball.sh
# Diagnostic (zero latency, but a new chunk only every PERIOD seconds):
#   PERIOD=0.36 LATENCIES=0.0 OUT=/content/results_ball_period bash scripts/vla/colab/eval_ball.sh
set -euo pipefail
CK=${CK:-/content/train_ball/checkpoints/020000/pretrained_model}
OUT=${OUT:-/content/results_ball}
EPISODES=${EPISODES:-100}
LATENCIES=${LATENCIES:-0.0 0.1 0.2 0.36}
PERIOD=${PERIOD:-}
TAG=+ball
EXTRA=()
if [ -n "$PERIOD" ]; then
  TAG="+ball/period$PERIOD"
  EXTRA=(--min-period "$PERIOD")
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p "$OUT"
for w in earth mars moon; do
  for L in $LATENCIES; do
    python scripts/vla/evaluate_vla.py --checkpoint "$CK" --worlds "$w" --episodes "$EPISODES" \
      --ball-state --tag "$TAG" --latencies "$L" --no-real --no-hier --skip-expert \
      --real-latency 0.348 "${EXTRA[@]}" --out "$OUT/${w}_L$L" > "$OUT/${w}_L$L.log" 2>&1 &
  done
done
wait
cat "$OUT"/*_L*/episodes.jsonl > "$OUT/episodes.jsonl"
echo "ball eval done"
