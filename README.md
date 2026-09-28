# Calibration Trap

Anonymous reproducibility package for the federated calibration experiments.

The package contains the training, evaluation, robustness, and post-processing code for CIFAR-10, CIFAR-100, and EMNIST ByClass, including Raw FedAvg, GlobalTS, LocalTS, FedTS, client-specific calibration evaluation, decomposition analyses, cross-fitted tail-selection sensitivity, label-heterogeneity and ECE-binning ablations, and the TailCal-v3/v4 intervention analyses.

The code retains the internal dataset key `FEMNIST` for backward compatibility with saved artifacts; manuscript-facing outputs refer to this dataset as EMNIST ByClass.



## layout

`src/` contains models, data partitioning, federated optimization, calibration, and evaluation. `experiments/` contains paper-facing entry points. `scripts/` contains shell wrappers.
The analysis package reproduces the paper-facing decomposition, Proposition 3 covariance identity, and M(gamma) weighting sweep from saved per-client CSV files. Cross-fitted tail-selection sensitivity and tail-membership stability are reproduced from saved `prediction_cache/*.npz` files containing client-level logits, labels, and FedAvg weights. Run `sh scripts/run_analysis.sh INPUT_DIR OUTPUT_DIR` after the primary experiment outputs and prediction caches exist. The Dirichlet ablation defaults to 150 training rounds. TailCal-v3 and TailCal-v4 each run a 50-round low-data second phase; v4 adds the frozen Phase-1 KL distillation term.

## Installation

Python 3.10+ is recommended.

pip install -r requirements.txt

### Main experiments

DEVICE=cuda sh scripts/run_main.sh DATA_ROOT OUTPUT_DIR

The default primary run evaluates seeds 42--46 on CIFAR-10, CIFAR-100, and EMNIST ByClass.

### Robustness analyses

DEVICE=cuda sh scripts/run_ablations.sh DATA_ROOT OUTPUT_DIR

This runs the Dirichlet, ECE-binning, and tail-fraction analyses.

### Decomposition and cross-fitting

sh scripts/run_analysis.sh OUTPUT_DIR ANALYSIS_OUTPUT_DIR

The decomposition and weighting analyses use the generated per-client CSV files. Cross-fitting additionally uses the prediction caches in OUTPUT_DIR/prediction_cache/.

### Training-time interventions

DEVICE=cuda sh scripts/run_interventions.sh DATA_ROOT OUTPUT_DIR

This evaluates TailCal-FL v3 across seeds 42--46 and TailCal-FL v4 for lambda in 0.1, 1, 5, and 10 across the same seeds.
