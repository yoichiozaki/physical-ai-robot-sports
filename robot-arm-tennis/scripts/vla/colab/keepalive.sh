#!/usr/bin/env bash
# Keep Colab sessions alive while background jobs run (idle runtimes get reclaimed).
#   bash scripts/vla/colab/keepalive.sh vla vla2
while true; do
  for s in "$@"; do
    echo 'print("alive")' | timeout 90 colab exec -s "$s" --timeout 30 > /dev/null 2>&1 || echo "$(date -u +%H:%M) keepalive failed: $s"
  done
  sleep 240
done