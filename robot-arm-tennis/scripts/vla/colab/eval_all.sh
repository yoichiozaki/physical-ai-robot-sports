#!/usr/bin/env bash
# Full closed-loop VLA evaluation on Colab: latency measured once (GPU otherwise idle), then the
# three planet worlds evaluated in parallel processes sharing the GPU.
#   CK=<checkpoint dir> OUT=/content/results_vla EPISODES=100 EXTRA="" bash scripts/vla/colab/eval_all.sh
set -euo pipefail
CK=${CK:-/content/train/checkpoints/020000/pretrained_model}
OUT=${OUT:-/content/results_vla}
EPISODES=${EPISODES:-100}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"
python scripts/vla/evaluate_vla.py --checkpoint "$CK" --out "$OUT/latency" --measure-only $EXTRA
LAT=$(python -c "import json; print(json.load(open('$OUT/latency/latency.json'))['mean_s'])")
echo "measured latency: $LAT s"
for w in earth mars moon; do
  python scripts/vla/evaluate_vla.py --checkpoint "$CK" --worlds "$w" --episodes "$EPISODES" \
    --real-latency "$LAT" --out "$OUT/$w" $EXTRA > "$OUT/$w.log" 2>&1 &
done
wait
cat "$OUT"/earth/episodes.jsonl "$OUT"/mars/episodes.jsonl "$OUT"/moon/episodes.jsonl > "$OUT/episodes.jsonl"
cp "$OUT/latency/latency.json" "$OUT/latency.json"
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "eval done"