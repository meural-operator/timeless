\# Benchmarking Neural Operators for Long-Horizon Dynamics



This repository provides a clean, reproducible benchmarking framework for

evaluating neural operator architectures (FNO, DeepONet, TurboNIGO) on

long-horizon rollout tasks for chaotic and dissipative PDEs.



\## Structure



\- `configs/` – YAML experiment configurations

\- `training/` – Training scripts (config-driven, resumable)

\- `models/` – Thin wrappers around reference implementations

\- `utils/` – Shared utilities (datasets, logging, checkpointing)

\- `external/` – Vendored reference implementations

\- `experiments/` – Generated outputs (gitignored)



\## Baselines



We vendor the reference implementations of:

\- Fourier Neural Operator

\- DeepONet



from https://github.com/lu-group/deeponet-fno (unmodified, commit pinned).



\## Usage



```bash

python training/train\_fno.py configs/fno\_bc.yaml

python training/train\_deeponet.py configs/deeponet\_bc.yaml



