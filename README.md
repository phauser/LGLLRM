# LGLLRM
Bayesian Generalized Linear Low-Rank Regression Models for Discrete Multivariate Longitudinal Outcomes (LGLLRM)

We propose the LGLLRM for the analysis of discrete, high-dimensional, multivariate longitudinal outcomes. The LGLLRM integrates three key methodologies: a low-rank matrix decomposition to approximate the relatively high dimensional regression coefficient matrix, a sparse factor model to capture the dependence among the multiple outcomes, and random effects to describe the dependence among the repeated responses. A sampling procedure combining the Gibbs sampler and Metropolis and Gamerman algorithms is employed to obtain posterior estimates of the regression coefficients and other model parameters.

This repository includes Rcpp code for the LGLLRM for binary and count outcomes, an R file to run the LGLLRM, and simulated data sets for binary and outcome outcomes.
