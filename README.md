# Decomposing the Calibration Trap

Anonymous code for the AISTATS 2027 submission *Decomposing the Calibration Trap: Client Heterogeneity and Sample-Weight Masking in Federated Learning*.

The repository covers training, evaluation, and post-processing for CIFAR-10, CIFAR-100, and EMNIST ByClass. It includes raw FedAvg and three post-hoc temperature-scaling baselines (GlobalTS, LocalTS, FedTS), client-specific calibration evaluation, the cwECE/uwECE decomposition, the cross-fitted tail-selection analysis, the label-heterogeneity and ECE-binning ablations, and the TailCal-FL v3/v4 intervention experiments.

**Dataset naming.** Code and saved artifacts use the internal key `FEMNIST` for backward compatibility. It refers to EMNIST ByClass partitioned with our Dirichlet and client-size procedure, not the writer-partitioned FEMNIST benchmark. The paper calls this dataset EMNIST.

## Repository layout

```
src/            models, data partitioning, federated optimization, calibration, evaluation
experiments/    entry points for the experiments reported in the paper
scripts/        shell wrappers for each stage below
```

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

## Reproducing the results

The stages run in order. Stage 3 reads the outputs of stage 1.

### 1. Primary experiments

```bash
DEVICE=cuda sh scripts/run_main.sh DATA_ROOT OUTPUT_DIR
```

Trains and evaluates seeds 42 through 46 on all three datasets. Writes per-client CSV files and prediction caches (`OUTPUT_DIR/prediction_cache/*.npz`, containing client-level logits, labels, and FedAvg weights).

### 2. Robustness ablations

```bash
DEVICE=cuda sh scripts/run_ablations.sh DATA_ROOT OUTPUT_DIR
```

Runs the Dirichlet label-heterogeneity sweep (150 training rounds by default), the ECE-binning comparison, and the tail-fraction sweep.

### 3. Decomposition and cross-fitting

```bash
sh scripts/run_analysis.sh OUTPUT_DIR ANALYSIS_OUTPUT_DIR
```

No training is required. The decomposition, the Proposition 3 covariance identity, and the M(γ) weighting sweep are computed from the per-client CSV files. The cross-fitted tail-selection analysis and tail-membership stability are computed from the prediction caches.

### 4. Training-time interventions

```bash
DEVICE=cuda sh scripts/run_interventions.sh DATA_ROOT OUTPUT_DIR
```

Runs TailCal-FL v3 and TailCal-FL v4 (λ ∈ {0.1, 1, 5, 10}) on seeds 42 through 46. Both add a 50-round second phase in which only low-data clients participate; v4 adds a KL distillation term toward the frozen Phase-1 model.

## Paper results

| Paper item | Stage |
|---|---|
| Table 1, Figure 2, Appendix Table 6 (decomposition, M(γ), per-seed diagnostics) | 3 |
| Table 2, Figure 3, Appendix B.2 (cross-fitting, tail membership) | 3 |
| Appendix Table 7 (post-hoc temperature scaling) | 1 |
| Table 3, Appendix Figure 4(a), Table 9, Figure 7 (ablations) | 2 |
| Appendix Tables 4–5, Figure 4(b) (interventions) | 4 |

## Released artifacts

Per-client outputs are provided for all five primary seeds (42 through 46). Original checkpoints and prediction caches are provided for seeds 42, 43, and 44, the seeds used in the cross-fitted analysis. The decomposition and weighting analyses for all five seeds can be reproduced from the released per-client outputs without retraining.

## Datasets

- CIFAR-10 and CIFAR-100 (Krizhevsky, 2009): https://www.cs.toronto.edu/~kriz/cifar.html
- EMNIST ByClass (Cohen et al., 2017): https://www.nist.gov/itl/products-and-services/emnist-dataset

Datasets are not redistributed. Use them under the terms of the original distributions.

## Hardware

Primary training was run on two NVIDIA Tesla T4 GPUs. Post-processing and analysis run on CPU.
