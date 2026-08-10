#!/usr/bin/env bash
# Nested cross-validation study.
#
# For each outer fold the four remaining folds are split again; the epoch and
# the raw/EMA choice are made on that inner split, and the outer fold is scored
# exactly once with the selected model. The reported number therefore carries
# no model-selection bias.
#
# Strongest model first, so an interrupted run still leaves the headline result.
set -uo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
PY=${PY:-.venv/bin/python}
EPOCHS=${EPOCHS:-12}
OUT=${OUT:-results}

step() { echo "=== $* ==="; }

step "deit3_small_patch16_224"
$PY -m gbc.cli --out "$OUT" train --backbone deit3_small_patch16_224 --epochs "$EPOCHS" --workers 0

step "efficientnet_b0 + second-order pooling"
$PY -m gbc.cli --out "$OUT" train --backbone efficientnet_b0 --epochs "$EPOCHS" --workers 0 --second-order

step "resnet50"
$PY -m gbc.cli --out "$OUT" train --backbone resnet50 --epochs "$EPOCHS" --workers 0

step "efficientnet_b0"
$PY -m gbc.cli --out "$OUT" train --backbone efficientnet_b0 --epochs "$EPOCHS" --workers 0

step "ensemble"
$PY -m gbc.cli --out "$OUT" ensemble

step "report"
$PY -m gbc.cli --out "$OUT" report
