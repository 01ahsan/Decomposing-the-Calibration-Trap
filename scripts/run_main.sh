#!/usr/bin/env sh
set -eu
DEVICE="${DEVICE:-cpu}"
python -m experiments.main_experiments --data-root "$1" --output-dir "$2" --device "$DEVICE"
