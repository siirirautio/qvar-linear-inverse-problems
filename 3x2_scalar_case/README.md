# 3×2 Linear Inverse Problem with Tikhonov Regularization

This folder contains the code and archived results for the 3×2 linear inverse problem presented in Section 6.2 of

> QVAR: A Quantum Variational Regularization Method for Linear Inverse Problems.

## Files

### Source code

- `scalar3x2_anneal_simulated.py` – simulation experiments
- `scalar3x2_anneal_q50.py` – experiments executed on the VTT Q50 quantum processor
- `generate_scalar3x2_figures.py` – generates the figures presented in the manuscript

### Archived results

Simulation results:

- `scalar3x2_results_simulated_T2.0_p4_eps0.05.npz`

Quantum hardware results:

- `scalar3x2_results_T2.0_p4_eps0.05.npz`

## Reproducibility

The archived result files contain the samples and processed data used to generate the figures reported in the manuscript.

The script `generate_scalar3x2_figures.py` can be used to generate the corresponding figures from the archived results.
