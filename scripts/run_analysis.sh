#!/usr/bin/env sh
set -eu
INPUT_DIR="$1"
OUTPUT_DIR="$2"
python -m analysis.decomposition --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
python -m analysis.weighting_gamma --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
python -m analysis.crossfit_sensitivity --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
