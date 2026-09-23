# NEXT

Official implementation of **NEXT**, a Neuro-Spectral architecture based on Exponential Time Differencing for solving forward and inverse PDE problems.

## Installation

This project uses Poetry.

```bash
poetry install
poetry shell
```

Alternatively,

```bash
poetry run python <script.py>
```

## Repository structure

```
basis/                 Spectral basis functions
numerical_solvers/     Reference PDE solvers
initial_condition/     Initial condition generators
integrators/           Time integration methods
models/                Neural network architectures
tests_NEXT/            Training and evaluation scripts
scripts/               Experiment scripts
data/                  Benchmark datasets
```

## Running experiments

Examples:

```bash
python tests_NEXT/heat1d_NEXT.py
python tests_NEXT/burgers1d_NEXT.py
```

The `scripts/` directory contains shell scripts reproducing the experiments reported in the paper:

| Script | Description |
|--------|-------------|
| `train_forward_problem.sh` | Forward problems (Table 1) |
| `inverse_burgers.sh` | Inverse problem in Burgers equation (Table 2) |
| `solver_ablation_heat.sh` | Ablation on ETD order (Appendix C.1, Table 6) |
| `solver_ablation_wave.sh` | Ablation on ETD order (Appendix C.1, Table 6) |
| `architecture_ablation.sh` | Architecture ablation (Appendix C.2, Table 7) |


