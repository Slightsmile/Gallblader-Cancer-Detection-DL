#!/usr/bin/env bash
# Full study. Order matters: the leakage ablation is the headline contribution,
# so it runs before any further backbone training and cannot be lost to an
# interrupted queue.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
EPOCHS=${EPOCHS:-12}
OUT=${OUT:-results}

# If a training process from a previous invocation is still running, wait it out
# rather than contending for the GPU.
while pgrep -f "gbc.cli --out $OUT train" >/dev/null; do sleep 30; done

step() { echo "=== $* ==="; }

step "leakage ablation"
$PY -m gbc.cli --out "$OUT" leakage-ablation --epochs 8 --cv-folds 3 --workers 0

step "shortcut baselines"
$PY -m gbc.cli --out "$OUT" shortcuts

for backbone in resnet50 deit3_small_patch16_224; do
  step "$backbone"
  $PY -m gbc.cli --out "$OUT" train --backbone "$backbone" --epochs "$EPOCHS" --workers 0
done

# Same backbone and schedule as the average-pooling run already completed, so the
# covariance-pooling head is the only variable.
step "efficientnet_b0 + second-order pooling"
$PY -m gbc.cli --out "$OUT" train --backbone efficientnet_b0 --epochs "$EPOCHS" \
  --workers 0 --second-order

step "ensemble"
$PY -m gbc.cli --out "$OUT" ensemble

step "report"
$PY -m gbc.cli --out "$OUT" report
