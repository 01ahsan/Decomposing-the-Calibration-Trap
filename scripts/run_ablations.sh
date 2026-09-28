#!/usr/bin/env sh
set -eu
DEVICE="${DEVICE:-cpu}"
python -m experiments.dirichlet_ablation --data-root "$1" --output-dir "$2" --device "$DEVICE"
python -m experiments.estimator_sensitivity --data-root "$1" --output-dir "$2"
python -m experiments.tail_fraction_sensitivity --data-root "$1" --output-dir "$2" --device "$DEVICE"
