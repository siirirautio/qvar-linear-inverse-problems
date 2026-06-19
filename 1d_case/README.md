# 1-Scalar Linear Inverse Problem with Tikhonov Regularization

This folder contains the code and archived results for the 1-scalar linear inverse problem presented in Section 6.1 of

> QVAR: A Quantum Variational Regularization Method for Linear Inverse Problems.

## Files

### Source code

- `scalar_anneal_simulated.py` – simulation experiments
- `scalar_anneal_q50.py` – experiments executed on the VTT Q50 quantum processor
- `generate_scalar1d_histograms.py` – generates the histograms presented in the manuscript

### Archived results

Simulation results:

- `samples_simu_T2_p2_eps0.04.json`
- `samples_simu_T100_p100_eps0.04.json`
- `samples_simu_T1000_p1000_eps0.04.json`

Quantum hardware results:

- `samples_q50_T2_p2_eps0.15.json`
- `samples_q50_T100_p100_eps0.15.json`
- `samples_q50_T200_p200_eps0.15.json`

## Reproducibility

The JSON files contain the archived measurement samples used to generate the figures reported in the manuscript.

The script `generate_scalar1d_histograms.py` can be used to generate the corresponding histograms from the archived results.
