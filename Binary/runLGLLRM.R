library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
library(RcppParallel)

nThreads = defaultNumThreads()

load("~/Binary/Case11.Rdata")
Rcpp::sourceCpp('~/Binary/LGLLRM.cpp')

# set q and r
r=3; q=2

# set tuning parameters for sampling (want acceptance rates 10-40%)
eE=12; eb=.0008; eU=.6; eV=.6; eD=.01; eL=.08

mod = LGLLRM(Y, X, Z, m, r=r, q=q, pb, 
              epsEta=eE, epsb=eb, epsU=eU, epsV=eV, epsD=eD, epsL=eL,  
              nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)
