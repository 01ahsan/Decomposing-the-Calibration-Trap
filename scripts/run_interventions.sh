#!/usr/bin/env sh
set -eu
python -m experiments.tailcal_v3 --data-root "$1" --output-dir "$2" --seed 42
python -m experiments.tailcal_v4 --data-root "$1" --output-dir "$2" --lambda-value 1
