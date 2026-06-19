# Optimization with ℓ¹-regularization in Haar-wavelet basis

This folder contains the code and archived results for the Haar-wavelet deconvolution problem with ℓ¹-regularization presented in Section 6.3 of

> QVAR: A Quantum Variational Regularization Method for Linear Inverse Problems.

## Files

### Source code

- `QUBO_haar.py` – constructs and solves the QUBO formulation of the Haar-wavelet deconvolution problem
- `generate_haar_figures.py` – generates the figures presented in the manuscript

### Archived results

Quantum hardware results:

- `samples_bits2_ncoeffs5_T4.0_p3_lambda0.05_eps0.03.json`
- `samples_bits3_ncoeffs5_T4.0_p3_lambda0.05_eps0.03.json`
- `samples_bits4_ncoeffs5_T4.0_p3_lambda0.05_eps0.03.json`

The result files correspond to different binary discretizations of the wavelet coefficients.

## Reproducibility

The JSON files contain the archived measurement samples used to generate the figures reported in the manuscript.

The script `generate_haar_figures.py` can be used to generate the corresponding figures from the archived results.
