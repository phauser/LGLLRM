library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
library(RcppParallel)

nThreads = defaultNumThreads()

load("~/Count/Case11.Rdata")
Rcpp::sourceCpp('~/Count/LGLLRM_count.cpp')

# set r (numbers of layers in decomposition) and q (number of latent factors)
r=3; q=2 

# set tuning parameters for sampling (want acceptance rates between 10-40%)
eE=8; eb=1; eU=.5; eV=.05; eD=.008; eL=.15

# Y is the outcome matrix (N x d)
# X is the design matrix for the regression coefficients (N x p)
# Z is the design matrix for the random effects (N x pb)
# m is a vector indicating the number of repeated measurements per subject (length=n, sum(m)=N)
mod = LGLLRM_count(Y, X, Z, m, r=r, q=q, pb=1, 
            epsEta=eE, epsb=eb, epsU=eU, epsV=eV, epsD=eD, epsL=eL,  
            nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)

