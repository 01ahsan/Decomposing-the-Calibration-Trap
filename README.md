# Calibration Trap

Anonymous reproducibility package for the federated calibration experiments.

The package contains the final training and evaluation pipeline for CIFAR-10, CIFAR-100, and FEMNIST, including Raw, GlobalTS, LocalTS, and FedTS; client-specific test-set evaluation; TailTrap; ECE-weighted TailCal-TS; size-corrected TailCal-FL; estimator sensitivity; tail-fraction sensitivity; and the CIFAR-100 Dirichlet ablation.

## Setup

```text
python -m venv .venv
python -m pip install -r requirements.txt
```

## Reproduce the main experiments

Supply a writable data directory and an output directory. Datasets are downloaded by torchvision when absent.

```text
python -m experiments.main_experiments --data-root DATA_DIR --output-dir OUTPUT_DIR
```

The default main configuration uses seeds 42 through 46, 200 client-specific evaluation examples per client drawn from the official test split, equal-width 15-bin ECE, and tail fraction 0.20. Use `--datasets CIFAR10 CIFAR100 FEMNIST` to select a subset and `--device cpu` for a CPU-only run.

## Reproduce robustness experiments

```text
python -m experiments.dirichlet_ablation --data-root DATA_DIR --output-dir OUTPUT_DIR
python -m experiments.estimator_sensitivity --data-root DATA_DIR --output-dir OUTPUT_DIR
python -m experiments.tail_fraction_sensitivity --data-root DATA_DIR --output-dir OUTPUT_DIR
python -m experiments.tailcal_v3 --data-root DATA_DIR --output-dir OUTPUT_DIR --seed 42
python -m experiments.tailcal_v4 --data-root DATA_DIR --output-dir OUTPUT_DIR --lambda-value 3
```

Training is deterministic up to backend-level numerical variation. Checkpoints and CSV files are written only below the supplied output directory.

## Evaluation protocols

`small_local_val` evaluates on each client's held-out local validation set. `large_client_eval` samples 200 examples per client from the official test split according to the realized client label mixture; samples may overlap across clients. The two protocols are reported separately.

## Repository layout

`src/` contains models, data partitioning, federated optimization, calibration, and evaluation. `experiments/` contains paper-facing entry points. `scripts/` contains shell wrappers. `results/` is reserved for generated outputs and contains no submitted results.
