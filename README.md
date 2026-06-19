# QVAR

Copyright (c) 2026 Siiri Rautio, Hjørdis Schlüter, Andreas Hauptmann, Babak Maboudi Afkham

This repository contains the source code, archived quantum hardware results, and plotting scripts required to reproduce the numerical experiments presented in the manuscript

> QVAR: A Quantum Variational Regularization Method for Linear Inverse Problems.

## Authors

Siiri Rautio¹, 
Hjørdis Schlüter²,
Andreas Hauptmann³⁴,
Babak Maboudi Afkham³

1. Department of Mathematics and Information Science, Josai University, Japan
2. Department of Mathematics and Statistics, University of Helsinki, Finland
3. Research Unit of Mathematical Sciences, University of Oulu, Finland
4. Department of Computer Science, University College London, United Kingdom

## Repository structure

- 1d_case/: Code and archived results for the scalar linear inverse problem with Tikhonov regularization presented in Section 6.1 of the manuscript.
- 3x2_scalar_case/: Code and archived results for the 3×2 linear inverse problem with Tikhonov regularization presented in Section 6.2 of the manuscript.
- haar_wavelet_case/: Code and archived results for the Haar-wavelet deconvolution problem with ℓ¹-regularization presented in Section 6.3 of the manuscript.

## Reproducibility

The repository contains all source code and archived experimental results required to reproduce the numerical experiments and figures presented in the manuscript. The archived quantum hardware results are provided to ensure reproducibility without requiring access to quantum hardware.

Each experiment folder contains:

- source code used to generate the results,
- archived measurement results,
- scripts used to generate the figures appearing in the manuscript.

## Software Requirements
The codes were developed and tested using Python 3 together with the following packages:
- NumPy
- SciPy
- Matplotlib
- Qiskit
- IQM Qiskit Provider
  
The experiments were executed using the software environment available through the LUMI supercomputer at the time of the experiments (April-June 2026).

## Citation

If you use this repository, please cite the associated manuscript.
A Zenodo DOI will be added upon archival of the repository.
