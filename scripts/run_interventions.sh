#!/usr/bin/env sh
set -eu

DEVICE="${DEVICE:-cpu}"
DATA_ROOT="$1"
OUT="$2"

python -m experiments.tailcal_v3 \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT" \
    --seeds 42 43 44 45 46 \
    --device "$DEVICE"

for LAMBDA in 0.1 1 5 10
do
    python -m experiments.tailcal_v4 \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUT" \
        --lambda-value "$LAMBDA" \
        --seeds 42 43 44 45 46 \
        --device "$DEVICE"
done