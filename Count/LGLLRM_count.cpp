#include <RcppDist.h>
#include <RcppArmadillo.h>
#include <RcppParallel.h>

// #include <cmath>
// #include <random>
// #include <string>
// #include <vector>

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppParallel)]]

inline double postDelta(std::size_t i, 
    const arma::mat &Y, const arma::mat &X,   const arma::mat &Z,
    const arma::mat &b, const arma::mat &eta, const arma::mat &lambda,
    const arma::mat &U, const arma::mat &V,   const arma::vec &delta,  double tau) {
  double lp = arma::as_scalar(-0.5*tau*pow(delta(i),2) + 
                delta(i)*U.col(i).t()*X.t()*Y*V.col(i) - 
                sum(sum(sum(exp(X*U*diagmat(delta)*V.t()+Z*b+eta*lambda.t())))));
  return lp;  
}

arma::vec UpdateDelta(std::size_t i, 
    const arma::mat &Y, const arma::mat &X,   const arma::mat &Z,
    const arma::mat &b, const arma::mat &eta, const arma::mat &lambda, 
    const arma::mat &U, const arma::mat &V,   const arma::vec &delta,  double tau, double eps) {
  arma::vec res(2);
  for (int i = 0; i < delta.size(); i++) {
    arma::vec delta_new = delta;
    delta_new(i) = arma::as_scalar(arma::randn(1)*eps+delta(i));
    double l0 = postDelta(i, Y, X, Z, b, eta, lambda, U, V, delta, tau);
    double l1 = postDelta(i, Y, X, Z, b, eta, lambda, U, V, delta_new, tau);
    
    // sample with acceptance probability min(1, exp(l1-l0))
    res(0) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0)));
    
    // return old value if acc=0 and new value if acc=1
    res(1) = res(0)*delta_new(i) + (1-res(0))*delta(i);
  }
  
  return res;
}

inline double postU(std::size_t j, 
    const arma::mat &Y, const arma::mat &X,   const arma::mat &Z,
    const arma::mat &b, const arma::mat &eta, const arma::mat &lambda,
    const arma::mat &U, const arma::mat &V,   const arma::vec &delta,  double tau) {
  double lp = arma::as_scalar(-0.5*tau*U.col(j).t()*U.col(j) + 
                delta(j)*U.col(j).t()*X.t()*Y*V.col(j) -
                sum(sum(sum(exp(X*U*diagmat(delta)*V.t()+Z*b+eta*lambda.t())))));
  return lp;  
}

struct updateU : public Worker {
  
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> Z;
  const RMatrix<double> b;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> U;
  const RMatrix<double> V;
  const RVector<double> delta;
  double tau;
  double eps;
  int j;
  int N;
  int n;
  int p;
  int d;
  int q;
  int pb;
  int r;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateU(const NumericMatrix Y,   const NumericMatrix X,   const NumericMatrix Z, 
          const NumericMatrix b,   const NumericMatrix eta, const NumericMatrix lambda,
          const NumericMatrix U,   const NumericMatrix V, const NumericVector delta, 
          double tau, double eps, int j, int N, int n, int p, int d, int q, int pb, int r, NumericMatrix res)
    : Y(Y), X(X), Z(Z), b(b), eta(eta), lambda(lambda), U(U), V(V), delta(delta), 
      tau(tau), eps(eps), j(j), N(N), n(n), p(p), d(d), q(q), pb(pb), r(r), res(res) {}
  
  arma::vec convertDelta(){
    RVector<double> d = delta;
    arma::vec VEC(d.begin(), r, false);
    return VEC;
  }
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), N, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), N, p, false);
    return MAT;
  }
  arma::mat convertZ(){
    RMatrix<double> z = Z;
    arma::mat MAT(z.begin(), N, n*pb, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), N, q, false);
    return MAT;
  }
  arma::mat convertb(){
    RMatrix<double> B = b;
    arma::mat MAT(B.begin(), n*pb, d, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertU(){
    RMatrix<double> u = U;
    arma::mat MAT(u.begin(), p, r, false);
    return MAT;
  }
  arma::mat convertV(){
    RMatrix<double> v = V;
    arma::mat MAT(v.begin(), d, r, false);
    return MAT;
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      arma::mat y = convertY();
      arma::mat x = convertX();
      arma::mat z = convertZ();
      arma::mat e = convertEta();
      arma::mat B = convertb();
      arma::mat l = convertLambda();
      arma::vec d = convertDelta();
      arma::mat u = convertU();
      arma::mat v = convertV();
      arma::mat u_new = u;
      
      u_new(i,j) = arma::as_scalar(arma::randn(1)*eps+u(i,j));
      double l0 = postU(j, y, x, z, B, e, l, u, v, d, tau);
      double l1 = postU(j, y, x, z, B, e, l, u_new, v, d, tau);
      
      // decide if we will accept w/prob=min(1, exp(l1-l0))
      res(i,0) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0)));
      
      // return old value if acc=0 and new value if acc=1
      res(i,1) = res(i,0)*u_new(i,j) + (1-res(i,0))*u(i,j);
    }
  }
};

arma::mat UpdateU(int j, 
    const NumericMatrix Y, const NumericMatrix X,   const NumericMatrix Z, 
    const NumericMatrix b, const NumericMatrix eta, const NumericMatrix lambda, 
    const NumericMatrix U, const NumericMatrix V,   const NumericVector delta,  
    double tau, double eps, int pb, int numThreads) {
  
  int N = Y.nrow();
  int n = b.nrow()/pb;
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();
  int r = delta.length();
  
  // allocate the matrix we will return
  NumericMatrix res(U.nrow(), 2);
  
  // create the worker
  updateU obj(Y, X, Z, b, eta, lambda, U, V, delta, tau, eps, j, N, n, p, d, q, pb, r, res);
  
  // call it with parallelFor
  // parallelFor(0, U.nrow(), obj, numThreads);
  parallelFor(0, U.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}

inline double postV(std::size_t j, 
    const arma::mat &Y,   const arma::mat &X,   const arma::mat &Z,
    const arma::mat &b,   const arma::mat &eta, const arma::mat &lambda,
    const arma::mat &U,   const arma::mat &V,   const arma::vec &delta,  double tau) {
  double lp = arma::as_scalar(-0.5*tau*V.col(j).t()*V.col(j) + 
                delta(j)*U.col(j).t()*X.t()*Y*V.col(j) -
                sum(sum(sum(exp(X*U*diagmat(delta)*V.t()+Z*b+eta*lambda.t())))));
  return lp;  
}

struct updateV : public Worker {
  
  // input matrix to read from
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> Z;
  const RMatrix<double> b;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> U;
  const RMatrix<double> V;
  const RVector<double> delta;
  double tau;
  double eps;
  int j;
  int N;
  int n;
  int p;
  int d;
  int q;
  int pb;
  int r;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateV(const NumericMatrix Y,   const NumericMatrix X,   const NumericMatrix Z, 
          const NumericMatrix b,   const NumericMatrix eta, const NumericMatrix lambda,
          const NumericMatrix U,   const NumericMatrix V, const NumericVector delta, 
          double tau, double eps, int j, int N, int n, int p, int d, int q, int pb, int r, NumericMatrix res)
    : Y(Y), X(X), Z(Z), b(b), eta(eta), lambda(lambda), U(U), V(V), delta(delta), 
      tau(tau), eps(eps), j(j), N(N), n(n), p(p), d(d), q(q), pb(pb), r(r), res(res) {}
  
  arma::vec convertDelta(){
    RVector<double> d = delta;
    arma::vec VEC(d.begin(), r, false);
    return VEC;
  }
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), N, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), N, p, false);
    return MAT;
  }
  arma::mat convertZ(){
    RMatrix<double> z = Z;
    arma::mat MAT(z.begin(), N, n*pb, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), N, q, false);
    return MAT;
  }
  arma::mat convertb(){
    RMatrix<double> B = b;
    arma::mat MAT(B.begin(), n*pb, d, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertU(){
    RMatrix<double> u = U;
    arma::mat MAT(u.begin(), p, r, false);
    return MAT;
  }
  arma::mat convertV(){
    RMatrix<double> v = V;
    arma::mat MAT(v.begin(), d, r, false);
    return MAT;
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      arma::mat y = convertY();
      arma::mat x = convertX();
      arma::mat z = convertZ();
      arma::mat e = convertEta();
      arma::mat B = convertb();
      arma::mat l = convertLambda();
      arma::vec d = convertDelta();
      arma::mat u = convertU();
      arma::mat v = convertV();
      arma::mat v_new = v;
      
      v_new(i,j) = arma::as_scalar(arma::randn(1)*eps+v(i,j));
      double l0 = postV(j, y, x, z, B, e, l, u, v,     d, tau);
      double l1 = postV(j, y, x, z, B, e, l, u, v_new, d, tau);
      
      // decide if we will accept w/prob=min(1, exp(l1-l0))
      res(i,0) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0)));
      
      // return old value if acc=0 and new value if acc=1
      res(i,1) = res(i,0)*v_new(i,j) + (1-res(i,0))*v(i,j);
    }
  }
};

arma::mat UpdateV(int j, 
    const NumericMatrix Y, const NumericMatrix X,   const NumericMatrix Z, 
    const NumericMatrix b, const NumericMatrix eta, const NumericMatrix lambda, 
    const NumericMatrix U, const NumericMatrix V,   const NumericVector delta,  
    double tau, double eps, int pb, int numThreads) {
  
  int N = Y.nrow();
  int n = b.nrow()/pb;
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();
  int r = delta.length();
  
  // allocate the matrix we will return
  NumericMatrix res(V.nrow(), 2);
  
  // create the worker
  updateV obj(Y, X, Z, b, eta, lambda, U, V, delta, tau, eps, j, N, n, p, d, q, pb, r, res);
  
  // call it with parallelFor
  // parallelFor(0, V.nrow(), obj, numThreads);
  parallelFor(0, V.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}

inline double postLambda(std::size_t i, 
    const arma::mat &Y,   const arma::mat &X,    const arma::mat &Z,
    const arma::mat &b,   const arma::mat &eta,  const arma::mat &lambda, 
    const arma::mat &U,   const arma::mat &V,    const arma::vec &delta,  const arma::mat &Dinvi) {
  double lp = arma::as_scalar(-0.5*lambda.row(i)*Dinvi*lambda.row(i).t() +
              lambda.row(i)*eta.t()*Y.col(i) -
             sum(sum(exp(X*U*arma::diagmat(delta)*V.row(i).t()+Z*b.col(i)+eta*lambda.row(i).t()))));
  return lp;  
}

struct updateLambda : public Worker {
  
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> Z;
  const RMatrix<double> b;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> U;
  const RMatrix<double> V;
  const RVector<double> delta;
  const RMatrix<double> tau;
  double eps;
  int N;
  int n;
  int p;
  int d;
  int q;
  int pb;
  int r;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateLambda(
      const NumericMatrix Y, const NumericMatrix X,   const NumericMatrix Z, 
      const NumericMatrix b, const NumericMatrix eta, const NumericMatrix lambda,
      const NumericMatrix U, const NumericMatrix V,   const NumericVector delta, 
      const NumericMatrix tau, double eps, int N, int n, int p, int d, int q, int pb, int r, NumericMatrix res)
    : Y(Y), X(X), Z(Z), b(b), eta(eta), lambda(lambda), U(U), V(V), delta(delta), 
      tau(tau), eps(eps), N(N), n(n), p(p), d(d), q(q), pb(pb), r(r), res(res) {}
  

  arma::mat convertTau(){
    RMatrix<double> Tau = tau;
    arma::mat MAT(Tau.begin(), d, q, false);
    return MAT;
  }
  arma::vec convertDelta(){
    RVector<double> d = delta;
    arma::vec VEC(d.begin(), r, false);
    return VEC;
  }
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), N, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), N, p, false);
    return MAT;
  }
  arma::mat convertZ(){
    RMatrix<double> z = Z;
    arma::mat MAT(z.begin(), N, n*pb, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), N, q, false);
    return MAT;
  }
  arma::mat convertb(){
    RMatrix<double> B = b;
    arma::mat MAT(B.begin(), n*pb, d, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertU(){
    RMatrix<double> u = U;
    arma::mat MAT(u.begin(), p, r, false);
    return MAT;
  }
  arma::mat convertV(){
    RMatrix<double> v = V;
    arma::mat MAT(v.begin(), d, r, false);
    return MAT;
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for(std::size_t j=0; j<lambda.ncol(); j++){
        arma::mat y = convertY();
        arma::mat x = convertX();
        arma::mat z = convertZ();
        arma::mat e = convertEta();
        arma::mat B = convertb();
        arma::mat l = convertLambda();
        arma::vec d = convertDelta();
        arma::mat u = convertU();
        arma::mat v = convertV();
        arma::mat t = convertTau();
        arma::mat l_new = l;
        arma::mat Dinvi(q,q, arma::fill::eye);
        Dinvi = arma::diagmat(t.row(i));
        if(i>j){l_new(i,j) = arma::as_scalar(arma::randn(1)*eps+l(i,j));}
        
        double Q = arma::log_normpdf(l(i,j), l_new(i,j), eps) - arma::log_normpdf(l_new(i,j), l(i,j), eps);
        
        double l0 = postLambda(i, y, x, z, B, e, l,     u, v, d, Dinvi);
        double l1 = postLambda(i, y, x, z, B, e, l_new, u, v, d, Dinvi);
        
        // decide if we will accept w/prob=min(1, exp(l1-l0))
        res(i,j) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0+Q)));
        
        // return old value if acc=0 and new value if acc=1
        res(i,j+l.n_cols) = res(i,j)*l_new(i,j) + (1-res(i,j))*l(i,j);
      }
    }
  }
};

arma::mat UpdateLambda(
    const NumericMatrix Y, const NumericMatrix X,   const NumericMatrix Z, 
    const NumericMatrix b, const NumericMatrix eta, const NumericMatrix lambda, 
    const NumericMatrix U, const NumericMatrix V,   const NumericVector delta,
    const NumericMatrix tau, double eps, int pb, int numThreads) {
  
  int N = Y.nrow();
  int n = b.nrow()/pb;
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();
  int r = delta.length();
  
  // allocate the matrix we will return
  NumericMatrix res(lambda.nrow(), 2*lambda.ncol());
  
  // create the worker
  updateLambda obj(Y, X, Z, b, eta, lambda, U, V, delta, tau, eps, N, n, p, d, q, pb, r, res);
  
  // call it with parallelFor
  // parallelFor(0, lambda.nrow(), obj, numThreads);
  parallelFor(0, lambda.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}

inline double postEta(std::size_t i, 
    const arma::mat &Y,  const arma::mat &X,   const arma::mat &Z,
    const arma::mat &b,  const arma::mat &eta, const arma::mat &lambda, 
    const arma::mat &U,  const arma::mat &V,   const arma::vec &delta,  double tau) {

   double lp = arma::as_scalar(-0.5*tau*eta.row(i)*eta.row(i).t()
                + Y.row(i)*lambda*eta.row(i).t()
                - sum(exp(X.row(i)*U*diagmat(delta)*V.t()+Z.row(i)*b+eta.row(i)*lambda.t())));
  return lp;  
}

struct updateEta : public Worker {
  
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> Z;
  const RMatrix<double> b;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> U;
  const RMatrix<double> V;
  const RVector<double> delta;
  double tau;
  double eps;
  int N;
  int n;
  int p;
  int d;
  int q;
  int r;
  int pb;
  std::string proposal;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateEta(std::string proposal,  
            const NumericMatrix Y, const NumericMatrix X,   const NumericMatrix Z, 
            const NumericMatrix b, const NumericMatrix eta, const NumericMatrix lambda, 
            const NumericMatrix U, const NumericMatrix V,   const NumericVector delta,  
            double tau, double eps, int N, int n, int p, int d, int q, int r, int pb, NumericMatrix res)
    : proposal(proposal), Y(Y), X(X), Z(Z), b(b), eta(eta), lambda(lambda), U(U), V(V), delta(delta), 
      tau(tau), eps(eps), N(N), n(n), p(p), d(d), q(q), r(r), pb(pb), res(res) {}
  
  arma::vec convertDelta(){
    RVector<double> d = delta;
    arma::vec VEC(d.begin(), r, false);
    return VEC;
  }
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), N, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), N, p, false);
    return MAT;
  }
  arma::mat convertZ(){
    RMatrix<double> z = Z;
    arma::mat MAT(z.begin(), N, n*pb, false);
    return MAT;
  }
  arma::mat convertB(){
    RMatrix<double> B = b;
    arma::mat MAT(B.begin(), n*pb, d, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), N, q, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertU(){
    RMatrix<double> u = U;
    arma::mat MAT(u.begin(), p, r, false);
    return MAT;
  }
  arma::mat convertV(){
    RMatrix<double> v = V;
    arma::mat MAT(v.begin(), d, r, false);
    return MAT;
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for(std::size_t j = 0; j < eta.ncol(); j++)
      {
        arma::mat y = convertY();
        arma::mat x = convertX();
        arma::mat z = convertZ();
        arma::mat b = convertB();
        arma::mat e = convertEta();
        arma::mat l = convertLambda();
        arma::vec delta = convertDelta();
        arma::mat u = convertU();
        arma::mat v = convertV();
        arma::mat e_new = e;
        double q;
        
        if(proposal=="normal"){
          e_new(i,j) = arma::as_scalar(arma::randn(1)*eps+e(i,j));
          q = arma::log_normpdf(e(i,j), e_new(i,j), eps) 
            - arma::log_normpdf(e_new(i,j), e(i,j), eps);
        }
        
        if(proposal=="Gamerman"){
          arma::vec W_ij0(d), W_ij1(d);
          arma::vec Y_XB0(d), Y_XB1(d);
          arma::mat I = arma::ones(1,d);
          
          for(int j0=0; j0<d; j0++){
            double eta_ij = arma::as_scalar(e.row(i)*l.row(j0).t());
            double theta_ij = arma::as_scalar(x.row(i)*u*arma::diagmat(delta)*v.row(j0).t()+z.row(i)*b.col(j0)+eta_ij);
            W_ij0(j0) = arma::as_scalar(exp(theta_ij));
            Y_XB0(j0) = eta_ij-1+exp(-theta_ij)*y(i,j0);
          }
          
          double c_i0 = arma::as_scalar(1.0/(tau+I*arma::diagmat(W_ij0)*I.t()));
          double m_i0 = arma::as_scalar(c_i0*I*arma::diagmat(W_ij0)*Y_XB0);
          
          e_new(i,j) = arma::as_scalar(arma::randn(1)*eps*sqrt(c_i0)+m_i0);
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = e(i,j);}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = e(i,j);}
          
          for(int j1=0; j1<d; j1++){
            double eta_ij = arma::as_scalar(e_new.row(i)*l.row(j1).t());
            double theta_ij = arma::as_scalar(x.row(i)*u*arma::diagmat(delta)*v.row(j1).t()+z.row(i)*b.col(j1)+eta_ij);
            W_ij1(j1) = arma::as_scalar(exp(theta_ij));
            Y_XB1(j1) = eta_ij-1+exp(-theta_ij)*y(i,j1);
          }
          
          double c_i1 = arma::as_scalar(1.0/(tau+I*arma::diagmat(W_ij1)*I.t()));
          double m_i1 = arma::as_scalar(c_i1*I*arma::diagmat(W_ij1)*Y_XB1);
          
          q = arma::log_normpdf(e(i,j),     m_i1, eps*pow(c_i1, 0.5))
            - arma::log_normpdf(e_new(i,j), m_i0, eps*pow(c_i0, 0.5));
        }
        
        double l0 = postEta(i, y, x, z, b, e,     l, u, v, delta, tau);
        double l1 = postEta(i, y, x, z, b, e_new, l, u, v, delta, tau);
        
        // decide if we will accept w/prob=min(1, exp(l1-l0))
        res(i,j) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0+q)));
        
        // return old value if acc=0 and new value if acc=1
        res(i,j+e.n_cols) = res(i,j)*e_new(i,j) + (1-res(i,j))*e(i,j);
      }
    }
  }
};

arma::mat UpdateEta(std::string proposal,
    const NumericMatrix Y,  const NumericMatrix X,   const NumericMatrix Z,  
    const NumericMatrix b,  const NumericMatrix eta, const NumericMatrix lambda, 
    const NumericMatrix U,  const NumericMatrix V,   const NumericVector delta,  
    double tau, double eps, int pb, int numThreads) {
  
  int N = Y.nrow();
  int n = b.nrow()/pb;
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();
  int r = delta.length();
  
  // allocate the matrix we will return
  NumericMatrix res(eta.nrow(), 2*eta.ncol());
  
  // create the worker
  updateEta obj(proposal, Y, X, Z, b, eta, lambda, U, V, delta, tau, eps, N, n, p, d, q, r, pb, res);
  
  // call it with parallelFor
  // parallelFor(0, eta.nrow(), obj, numThreads);
  parallelFor(0, eta.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}


// [[Rcpp::export]]
double postb(std::size_t i, std::size_t k, 
             const arma::mat &Y,  const arma::mat &X,   const arma::mat &Z,
             const arma::mat &b,  const arma::mat &eta, const arma::mat &lambda, 
             const arma::mat &U,  const arma::mat &V,   const arma::vec &delta,  
             const arma::vec &m,  double tau) {
  int n = m.n_elem;
  int pb = b.n_rows/n;
  
  int ind1=0, ind2=0;
  
  if(i==0){
    ind1 = 0;
    ind2 = m[0]-1;
  }
  else{
    ind1 = sum(m.subvec(0, i-1));
    ind2 = sum(m.subvec(0, i))-1;
  }
  
  arma::mat b_ik  = b.submat(  i*pb, k,    i*pb+pb-1, k);
  arma::mat Y_ik  = Y.submat(  ind1, k,    ind2,      k);
  arma::mat Z_i   = Z.submat(  ind1, i*pb, ind2,      i*pb+pb-1);
  arma::mat X_i   = X.submat(  ind1, 0,    ind2,      X.n_cols-1);
  arma::mat eta_i = eta.submat(ind1, 0,    ind2,      eta.n_cols-1);
  
  double lp = arma::as_scalar(-0.5*tau*b_ik.t()*b_ik + Y_ik.t()*Z_i*b_ik - 
                               sum(exp(X_i*U*diagmat(delta)*V.row(k).t()+Z_i*b_ik+eta_i*lambda.row(k).t())));
  return lp;  
}

struct updateb : public Worker {
  
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> Z;
  const RMatrix<double> b;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> U;
  const RMatrix<double> V;
  const RVector<double> delta;
  const RVector<double> m;
  const RVector<double> tau;
  double eps;
  int N;
  int n;
  int p;
  int d;
  int q;
  int r;
  int pb;
  std::string proposal;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateb(std::string proposal, 
          const NumericMatrix Y, const NumericMatrix X,   const NumericMatrix Z, 
          const NumericMatrix b, const NumericMatrix eta, const NumericMatrix lambda, 
          const NumericMatrix U, const NumericMatrix V,   const NumericVector delta,  
          const NumericVector m, const NumericVector tau, double eps, 
          int N, int n, int p, int d, int q, int r, int pb, NumericMatrix res)
    :  proposal(proposal), Y(Y), X(X), Z(Z), b(b), eta(eta), lambda(lambda), U(U), V(V), delta(delta), m(m),
      tau(tau), eps(eps), N(N), n(n), p(p), d(d), q(q), r(r), pb(pb), res(res) {}
  
  arma::vec convertDelta(){
    RVector<double> d = delta;
    arma::vec VEC(d.begin(), r, false);
    return VEC;
  }
  arma::vec convertM(){
    RVector<double> M = m;
    arma::vec VEC(M.begin(), n, false);
    return VEC;
  }
  arma::vec convertTau(){
    RVector<double> Tau = tau;
    arma::vec VEC(Tau.begin(), d, false);
    return VEC;
  }
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), N, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), N, p, false);
    return MAT;
  }
  arma::mat convertZ(){
    RMatrix<double> z = Z;
    arma::mat MAT(z.begin(), N, n*pb, false);
    return MAT;
  }
  arma::mat convertB(){
    RMatrix<double> B = b;
    arma::mat MAT(B.begin(), n*pb, d, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), N, q, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertU(){
    RMatrix<double> u = U;
    arma::mat MAT(u.begin(), p, r, false);
    return MAT;
  }
  arma::mat convertV(){
    RMatrix<double> v = V;
    arma::mat MAT(v.begin(), d, r, false);
    return MAT;
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for(std::size_t j = 0; j < b.ncol(); j++)
      {
        arma::mat y = convertY();
        arma::mat x = convertX();
        arma::mat z = convertZ();
        arma::mat b = convertB();
        arma::mat e = convertEta();
        arma::mat l = convertLambda();
        arma::vec delta = convertDelta();
        arma::vec M = convertM();
        arma::vec Tau = convertTau();
        arma::mat u = convertU();
        arma::mat v = convertV();
        arma::mat b_new = b;
        double q;
        
        int ind1=0, ind2=0;
        
        if(i==0){
          ind1 = 0;
          ind2 = m[0]-1;
        }
        else{
          ind1 = sum(M.subvec(0, i-1));
          ind2 = sum(M.subvec(0, i))-1;
        }
        
        if(proposal=="normal"){
          b_new(i,j) = arma::as_scalar(arma::randn(1)*eps+b(i,j));
          q = arma::log_normpdf(b(i,j), b_new(i,j), eps) 
            - arma::log_normpdf(b_new(i,j), b(i,j), eps);
        }
        
        if(proposal=="Gamerman"){
          arma::vec W_ij0(d), W_ij1(d);
          arma::vec Y_XB0(d), Y_XB1(d);
          arma::mat I = arma::ones(1,d);
          
          for(int j0=0; j0<d; j0++){
            arma::mat b_ik = b.submat(i*pb, j0,   i*pb+pb-1, j0);
            arma::mat Y_ik = y.submat(ind1, j0,   ind2,      j0);
            arma::mat Z_i  = z.submat(ind1, i*pb, ind2,      i*pb+pb-1);
            arma::mat X_i  = x.submat(ind1, 0,    ind2,      x.n_cols-1);
            arma::mat e_i  = e.submat(ind1, 0,    ind2,      e.n_cols-1);
            
            double eta_ij = arma::as_scalar(sum(Z_i*b_ik));
            double theta_ij = arma::as_scalar(sum(X_i*u*arma::diagmat(delta)*v.row(j0).t()+eta_ij+e_i*l.row(j0).t()));
            W_ij0(j0) = arma::as_scalar(exp(theta_ij));
            Y_XB0(j0) = eta_ij-1+exp(-theta_ij)*y(i,j0);
          }
          
          double c_i0 = arma::as_scalar(1.0/(Tau(j)+I*arma::diagmat(W_ij0)*I.t()));
          double m_i0 = arma::as_scalar(c_i0*I*arma::diagmat(W_ij0)*Y_XB0);
          
          b_new(i,j) = arma::as_scalar(arma::randn(1)*eps*sqrt(c_i0)+m_i0);
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(b_new(i,j)>1000 | b_new(i,j)<-1000){b_new(i,j) = b(i,j);}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(b_new(i,j))){b_new(i,j) = b(i,j);}
          
          for(int j1=0; j1<d; j1++){
            arma::mat b_ik = b.submat(i*pb, j1,   i*pb+pb-1, j1);
            arma::mat Y_ik = y.submat(ind1, j1,   ind2,      j1);
            arma::mat Z_i  = z.submat(ind1, i*pb, ind2,      i*pb+pb-1);
            arma::mat X_i  = x.submat(ind1, 0,    ind2,      x.n_cols-1);
            arma::mat e_i  = e.submat(ind1, 0,    ind2,      e.n_cols-1);
            
            double eta_ij = arma::as_scalar(sum(Z_i*b_ik));
            double theta_ij = arma::as_scalar(sum(X_i*u*arma::diagmat(delta)*v.row(j1).t()+eta_ij+e_i*l.row(j1).t()));
            W_ij0(j1) = arma::as_scalar(exp(theta_ij));
            Y_XB0(j1) = eta_ij-1+exp(-theta_ij)*y(i,j1);
          }
          
          double c_i1 = arma::as_scalar(1.0/(Tau(j)+I*arma::diagmat(W_ij1)*I.t()));
          double m_i1 = arma::as_scalar(c_i1*I*arma::diagmat(W_ij1)*Y_XB1);
          
          q = arma::log_normpdf(b(i,j),     m_i1, eps*pow(c_i1, 0.5))
            - arma::log_normpdf(b_new(i,j), m_i0, eps*pow(c_i0, 0.5));
        }
        
        double l0 = postb(i, j, y, x, z, b,     e, l, u, v, delta, M, Tau(j));
        double l1 = postb(i, j, y, x, z, b_new, e, l, u, v, delta, M, Tau(j));
          
        // decide if we will accept w/prob=min(1, exp(l1-l0))
        res(i,j) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0+q)));
        
        // return old value if acc=0 and new value if acc=1
        res(i,j+b.n_cols) = res(i,j)*b_new(i,j) + (1-res(i,j))*b(i,j);
      }
    }
  }
};

// [[Rcpp::export]]
arma::mat Updateb(std::string proposal, 
                  const NumericMatrix Y, const NumericMatrix X,   const NumericMatrix Z,  
                  const NumericMatrix b, const NumericMatrix eta, const NumericMatrix lambda, 
                  const NumericMatrix U, const NumericMatrix V,   const NumericVector delta,  
                  const NumericVector m, const NumericVector tau, double eps, int pb, int numThreads) {
  int N = Y.nrow();
  int n = b.nrow()/pb;
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();
  int r = delta.length();
  
  // allocate the matrix we will return
  NumericMatrix res(b.nrow(), 2*b.ncol());
  
  // create the worker
  updateb obj(proposal, Y, X, Z, b, eta, lambda, U, V, delta, m, tau, eps, N, n, p, d, q, r, pb, res);
  
  // call it with parallelFor
  parallelFor(0, b.nrow(), obj, numThreads);
  
  return as<arma::mat>(wrap(res));
}


// [[Rcpp::export]]
List LGLLRM_count(const arma::mat &Y, const arma::mat &X, const arma::mat &Z, 
                  const arma::vec & m, int & r, int & q, int pb, 
                  double & epsEta, double & epsb, double & epsU, double & epsV, 
                  double & epsD, double & epsL,
                  int & nBurnin, int & nCollect, int & thin, int numThreads){
  std::string proposal = "Gamerman";

  // get dimensions
  int N = X.n_rows, n=Z.n_cols/pb, p = X.n_cols, d = Y.n_cols;
  
  // initialize parameters for random effects
  mat b         = zeros(n*pb, d);
  double alpha  = 0.01;
  vec taub   = rgamma(d, alpha, 1.0/alpha); 
  vec varb   = 1.0/taub;
  vec rate;
  
  // initialize parameters for decomposition
  mat U          = ones(p, r)*0.1;
  mat V          = ones(d, r)*0.1;
  vec delta; delta.ones(r);
  
  double a0   = 0.00001;
  double tauU = 1.0/p;
  double tauV = 1.0/d;
  double tauD = 1.0/r;

  // initialize parameters for latent factor model
  mat eta        = zeros(N, q);
  double alpha1  = 0.01;
  double tauEta   = rgamma(1, alpha1, 1.0/alpha1)(0); 
  double varEta   = 1.0/tauEta;
  
  mat lambda     = rnorm(d*q, 0, 3);
  lambda.reshape(d,q);
  for(int i=0; i<d; i++){
    for(int j=0; j<q; j++){
      if(i==j){lambda(i,j)=1;}
      if(j>i){lambda(i,j)=0;}
    }
  }
  mat E = eta*lambda.t();
  
  double nu      = 3.0;     // gamma shape & rate hyperparameter for phi_{jh}
  double a_psi1  = 2.0;     // gamma shape hyperparameter for psi_1
  double b_psi1  = 1.0;     // gamma rate = 1/scale hyperparameter for psi_1
  double a_psi2  = 3.0;     // gamma shape hyperparameter for psi_g, g >= 2
  double b_psi2  = 1.0;     // gamma rate=1/scale hyperparameter for psi_g
  
  mat phi        = rgamma(d*q, nu/2.0, nu/2.0);  // local shrinkage parameter
  phi.reshape(d,q);                              
  mat psi1       = rgamma(1,   a_psi1, b_psi1);
  mat psi2       = rgamma(q-1, a_psi2, b_psi2);
  vec psi        = join_cols(psi1, psi2);        // global shrinkage coefficients multipliers
  vec tau        = cumprod(psi);                 // global shrinkage parameter (q x 1)
  mat Dinv       = phi%repmat(tau.t(), d, 1.0);  // D inverse (since we didn't invert tau and phi), d x q
  
  // for thinning
  int runs       = (nCollect+nBurnin)*thin;
  int totalIter  = 0;                        // counter for total iterations
  int idx        = 0;                        // counter for thinned iterations
  
  // initialize outputs
  mat SumB         = zeros(p, d);
  mat SumSB        = zeros(p, d);
  mat SumU         = zeros(p, r);
  mat SumV         = zeros(d, r);
  mat Sumdelta     = zeros(r, 1);
  mat Sumb         = zeros(n*pb, d);
  mat SumLambda    = zeros(d, q);
  mat SumEta       = zeros(N, q);
  mat SumE         = zeros(N,d);
  vec Sumvarb(d);
  double SumvarEta = 0;

  vec  Output_dev(nCollect);
  
  cube Output_B(p, d, nCollect);
  cube Output_lambda(d, q, nCollect);
  mat  Output_varb(d, nCollect);
  vec  Output_varEta(nCollect);

  mat AR_B          = zeros(p, d);
  mat SumAR_B       = zeros(p, d);
  mat AR_U          = zeros(p, r);
  mat SumAR_U       = zeros(p, r);
  mat AR_V          = zeros(d, r);
  mat SumAR_V       = zeros(d, r);
  mat AR_delta      = zeros(r, 1);
  mat SumAR_delta   = zeros(r, 1);
  mat AR_b          = zeros(n*pb, d);
  mat SumAR_b       = zeros(n*pb, d);
  mat AR_eta        = zeros(N, q);
  mat SumAR_eta     = zeros(N, q);
  mat AR_lambda     = zeros(d, q);
  mat SumAR_lambda  = zeros(d, q);
  mat MH_U, MH_V, MH_b, MH_eta, MH_lambda;
  vec MH_delta;
  
  // start the sampling
  for(int iter=0; iter<runs; iter++){
    // count the total number of iterations - want (nBurnin+nCollect)*thin total
    totalIter = iter;
    
    // B = U Delta V^T
    mat B = U*diagmat(delta)*V.t();

    for(int l=0; l<r; l++){
    // Step 1: Update U
    MH_U = UpdateU(l, as<NumericMatrix>(wrap(Y)), as<NumericMatrix>(wrap(X)),   as<NumericMatrix>(wrap(Z)),
                      as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(eta)), as<NumericMatrix>(wrap(lambda)),
                      as<NumericMatrix>(wrap(U)), as<NumericMatrix>(wrap(V)),   as<NumericVector>(wrap(delta)),
                      tauU, epsU, pb, numThreads);
    U.col(l)    = MH_U.col(1);
    AR_U.col(l) = MH_U.col(0);

    // Step 2: Update V
    MH_V = UpdateV(l, as<NumericMatrix>(wrap(Y)),   as<NumericMatrix>(wrap(X)),   as<NumericMatrix>(wrap(Z)),
                      as<NumericMatrix>(wrap(b)),   as<NumericMatrix>(wrap(eta)), as<NumericMatrix>(wrap(lambda)),
                      as<NumericMatrix>(wrap(U)),   as<NumericMatrix>(wrap(V)),   as<NumericVector>(wrap(delta)),
                      tauV, epsV, pb, numThreads);
    V.col(l)   = MH_V.col(1);
    AR_V.col(l) = MH_V.col(0);

    // Step 3: Update delta
    MH_delta = UpdateDelta(l, Y, X, Z, b, eta, lambda, U, V, delta, tauD, epsD);
    delta(l) = MH_delta(1);
    AR_delta(l) = MH_delta(0);
  }

    // Step 3.5: Update B
    B = U*diagmat(delta)*V.t();
    
    
    // Step 4: Update b
    MH_b = Updateb(proposal, as<NumericMatrix>(wrap(Y)), as<NumericMatrix>(wrap(X)),   as<NumericMatrix>(wrap(Z)),
                             as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(eta)), as<NumericMatrix>(wrap(lambda)),
                             as<NumericMatrix>(wrap(U)), as<NumericMatrix>(wrap(V)),   as<NumericVector>(wrap(delta)),
                             as<NumericVector>(wrap(m)), as<NumericVector>(wrap(taub)), epsb, pb, numThreads);

    b = MH_b.submat(0, d, b.n_rows-1, 2*d-1);
    AR_b = MH_b.submat(0, 0, b.n_rows-1, d-1);
    
    
    // Step 5: Update tau_b
    for(int j=0; j<d; j++){
       rate = 1.0/alpha + 0.5*sum(sum(pow(b.col(j),2)));
       taub(j) = rgamma(1, 0.5*(n*pb + alpha), 1.0/rate(0))(0);
     }
    varb = 1.0/taub;
    
    
    // Step 6: Update eta
    MH_eta = UpdateEta(proposal, as<NumericMatrix>(wrap(Y)), as<NumericMatrix>(wrap(X)),   as<NumericMatrix>(wrap(Z)),
                                 as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(eta)), as<NumericMatrix>(wrap(lambda)),
                                 as<NumericMatrix>(wrap(U)),   as<NumericMatrix>(wrap(V)), as<NumericVector>(wrap(delta)),
                                 tauEta, epsEta, pb, numThreads);
    eta    = MH_eta.submat(0, q, eta.n_rows-1, 2*q-1);
    AR_eta = MH_eta.submat(0, 0, eta.n_rows-1, q-1);

    // Step 7: Update tau_eta
    rate   = 1.0/alpha1 + 0.5*sum(sum(pow(eta,2)));
    tauEta = rgamma(1, 0.5*(N*q+ alpha1), 1.0/rate(0))(0); // or nN/2? Nq/2
    varEta = 1.0/tauEta;

    // Step 8: Update lambda d x q
    MH_lambda = UpdateLambda(as<NumericMatrix>(wrap(Y)), as<NumericMatrix>(wrap(X)),   as<NumericMatrix>(wrap(Z)),
                                     as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(eta)), as<NumericMatrix>(wrap(lambda)),
                                     as<NumericMatrix>(wrap(U)), as<NumericMatrix>(wrap(V)),as<NumericVector>(wrap(delta)),
                                     as<NumericMatrix>(wrap(Dinv)), epsL, pb, numThreads);
    lambda = MH_lambda.submat(0, q, d-1, 2*q-1);
    AR_lambda = MH_lambda.submat(0, 0, d-1, q-1);

    // Step 9: Update phi  d x q
    mat phi_scale = 0.5*nu + 0.5*pow(lambda, 2)%repmat(tau.t(), d, 1);
    for(int j=0; j<d; ++j){for(int h=0; h<q; ++h){phi(j,h) = rgamma(1, 0.5*nu + 0.5, 1.0/phi_scale(j,h))(0);}}
      
    // Step 10: Update psi
    mat phi_lam         = sum(phi%pow(lambda, 2));
    double phi_lam_tau  = arma::as_scalar(tau.t()*phi_lam.t()); 
    double b_psi        = b_psi1 + 0.5*(1.0/psi(0))*phi_lam_tau;
    psi(0)              = rgamma(1, a_psi1 + 0.5*d*q, 1.0/b_psi)(0);
    tau                 = cumprod(psi);
      
    for(int j=1; j<q; ++j){
       double a_psi = a_psi2 + 0.5*d*(q-j);
       vec temp1    = (tau.t()%phi_lam).t();
       double b_psi = b_psi1 + 0.5*(1.0/psi(j))*accu(temp1.subvec(j,q-1));
       psi(j)       = rgamma(1, a_psi, 1.0/b_psi)(0);
       tau          = cumprod(psi);
    }
      
     // Step 11: Update Dinv
     Dinv = phi%repmat(tau.t(), d, 1);
 
    
    E = eta*lambda.t();
    
    vec temp(d);
    for(int i=0; i<d; i++){
        arma::mat Dinvi(q,q, arma::fill::eye);
        Dinvi = arma::diagmat(Dinv.row(i));
        temp(i) = arma::as_scalar(lambda.row(i)*Dinvi*lambda.row(i).t());
    }
    
    mat ETA = X*B+Z*b+eta*lambda.t();
    mat ll10 = Y.t()*ETA;
    double ll20 = arma::as_scalar(sum(sum(exp(ETA))));
    double dev = -2*(sum(ll10.diag())-ll20);
    
    // Collect samples
    if((iter+1) > (nBurnin*thin)){
      if(iter % thin == 0){
        Output_B.slice(idx)       = B;
        Output_lambda.slice(idx)  = lambda;
        Output_varb.col(idx)      = 1.0/taub;
        Output_varEta(idx)        = arma::as_scalar(varEta);
        Output_dev(idx)           = dev;
        
        SumAR_U       += AR_U; 
        SumAR_V       += AR_V;
        SumAR_delta   += AR_delta;
        SumAR_b       += AR_b;
        SumAR_eta     += AR_eta;
        SumAR_lambda  += AR_lambda;
        SumB          += B;
        Sumdelta      += delta;
        Sumb          += b;
        SumEta        += eta;
        SumE          += E;
        Sumvarb       += varb;
        SumvarEta     += arma::as_scalar(varEta);
        SumLambda     += lambda;

        idx = idx+1;
      }
    }
  }
  
  // Output
  List Output;
  int N0 = nCollect;
  Output["dev"]             = Output_dev;
  Output["B"]               = Output_B;
  Output["varb"]            = Output_varb;
  Output["varEta"]          = Output_varEta;
  Output["lambda"]          = Output_lambda;

  Output["PostB"]           = SumB/N0;
  Output["PostDelta"]       = Sumdelta/N0;
  Output["Postb"]           = Sumb/N0;
  Output["PostEta"]         = SumEta/N0;
  Output["PostE"]           = SumE/N0;
  Output["Postvarb"]        = Sumvarb/N0;
  Output["PostvarEta"]      = SumvarEta/N0;
  Output["PostLambda"]      = SumLambda/N0;
  
  Output["AR_U"]            = SumAR_U/N0;
  Output["AR_V"]            = SumAR_V/N0;
  Output["AR_delta"]        = SumAR_delta/N0;
  Output["AR_b"]            = mean(SumAR_b/N0);
  Output["AR_eta"]          = mean(SumAR_eta/N0);
  Output["AR_lambda"]       = SumAR_lambda/N0;
  
  return Output;
}

// [[Rcpp::export]]
List PerformGLMM(mat Y, mat X,  mat Z, mat B, mat E, mat b, int r, vec dev0){
  mat ll1 = Y.t()*(X*B+E);
  double ll2 = arma::as_scalar(sum(sum(exp(X*B+E))));
  double dev = -2*(sum(ll1.diag()) - ll2); // deviance at the posterior mean
  double pd = mean(dev0)-dev; 
  double DIC = 2*pd+dev;
  return List::create(Named("DIC")=DIC,
                      Named("pd")=pd,
                      Named("dhat")=dev);
}