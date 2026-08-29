#!/usr/bin/env sh
set -eu
python -m experiments.main_experiments --data-root "$1" --output-dir "$2"
