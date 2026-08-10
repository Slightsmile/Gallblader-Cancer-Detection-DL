#!/usr/bin/env bash
# Full cross-validated study: one leakage-free 5-fold run per backbone, then ensembling.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
EPOCHS=${EPOCHS:-12}
OUT=${OUT:-results}

for backbone in efficientnet_b0 resnet50 deit3_small_patch16_224; do
  echo "=== $backbone ==="
  $PY -m gbc.cli --out "$OUT" train --backbone "$backbone" --epochs "$EPOCHS" --workers 0
done

echo "=== ensemble ==="
$PY -m gbc.cli --out "$OUT" ensemble

echo "=== leakage ablation ==="
$PY -m gbc.cli --out "$OUT" leakage-ablation --epochs 8 --cv-folds 3 --workers 0
