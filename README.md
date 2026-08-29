# Calibration Trap

Anonymous reproducibility package for the federated calibration experiments.

The package contains the final training and evaluation pipeline for CIFAR-10, CIFAR-100, and FEMNIST, including Raw, GlobalTS, LocalTS, and FedTS; client-specific test-set evaluation; TailTrap; ECE-weighted TailCal-TS; size-corrected TailCal-FL; estimator sensitivity; tail-fraction sensitivity; and the CIFAR-100 Dirichlet ablation.



## layout

`src/` contains models, data partitioning, federated optimization, calibration, and evaluation. `experiments/` contains paper-facing entry points. `scripts/` contains shell wrappers. 
