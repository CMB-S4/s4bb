# s4bb

This is a CMB-S4 focused python reimplementation of the BICEP multi-component cross-spectral likelihood.

Not yet ready for use -- still missing major components. *C Bischoff, 2024-09-10*

## To-do list

* Power spectrum estimation
  - Basic healpy tools
  - NaMaster
* Bandpower covariance matrix
  - Knox formula bandpower covariance matrix
  - Bandpower covariance matrix derived from signal and noise sims
* Likelihood calculation
  - Foreground models
* Exercise code on DC11 simulations; use this to generate examples
* Think about which classes need copy() methods
