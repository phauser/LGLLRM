library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
library(RcppParallel)

nThreads = defaultNumThreads()

load("~/Count/Case11.Rdata")
Rcpp::sourceCpp('~/LGLLRM_count.cpp')

# set r (numbers of layers in decomposition) and q (number of latent factors)
r=3; q=2 

# set tuning parameters for sampling (want acceptance rates between 10-40%)
eE=8; eb=1; eU=.5; eV=.05; eD=.008; eL=.15

mod = LGLLRM_count(Y, X, Z, m, pb=1, 
            epsEta=eE, epsb=eb, epsU=eU, epsV=eV, epsD=eD, epsL=eL,  
            r=r, q=q, nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)

