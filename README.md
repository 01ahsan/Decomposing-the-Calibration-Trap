# Decomposing the Calibration Trap

Anonymous code for federated calibration experiments on CIFAR-10, CIFAR-100, and EMNIST ByClass: raw FedAvg, post-hoc temperature scaling (GlobalTS, LocalTS, FedTS), client-specific calibration evaluation, the cwECE/uwECE decomposition, cross-fitted tail selection, ablations, and the TailCal-FL v3/v4 interventions.

**Dataset naming.** Code and saved artifacts use the internal key `FEMNIST` for EMNIST ByClass partitioned with our Dirichlet and client-size procedure. This is not the writer-partitioned FEMNIST benchmark.

## Layout

```
src/            models, data partitioning, federated optimization, calibration, evaluation
experiments/    experiment entry points
scripts/        shell wrappers for each stage below
```

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Running

Stages run in order; stage 3 reads the outputs of stage 1.

**1. Primary experiments**

```bash
DEVICE=cuda sh scripts/run_main.sh DATA_ROOT OUTPUT_DIR
```

Trains and evaluates seeds 42 through 46 on all three datasets. Writes per-client CSV files and prediction caches (`OUTPUT_DIR/prediction_cache/*.npz` with client-level logits, labels, and FedAvg weights).

**2. Ablations**

```bash
DEVICE=cuda sh scripts/run_ablations.sh DATA_ROOT OUTPUT_DIR
```

Dirichlet label-heterogeneity sweep (150 rounds by default), ECE-binning comparison, and tail-fraction sweep.

**3. Decomposition and cross-fitting**

```bash
sh scripts/run_analysis.sh OUTPUT_DIR ANALYSIS_OUTPUT_DIR
```

No training. The decomposition, covariance identity, and M(γ) sweep use the per-client CSV files. Cross-fitting and tail-membership stability use the prediction caches. The released caches cover seeds 42, 43, and 44.

**4. Training-time interventions**

```bash
DEVICE=cuda sh scripts/run_interventions.sh DATA_ROOT OUTPUT_DIR
```

TailCal-FL v3 and v4 (λ ∈ {0.1, 1, 5, 10}) on seeds 42 through 46. Both add a 50-round phase in which only low-data clients train; v4 adds a KL term toward the frozen Phase-1 model.

## Datasets

- CIFAR-10 / CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- EMNIST ByClass: https://www.nist.gov/itl/products-and-services/emnist-dataset

Datasets are not redistributed; use them under the terms of the original distributions.
