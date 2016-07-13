#include <Rcpp>
#include <cmath>
#include <vector>
#include <cmath> //not sure this is necessary
#include <algorithm>

// [[Rcpp::plugins(openmp)]]

// For Rcpp+OpenMP parallelization, check: https://wbnicholson.wordpress.com/2014/07/10/parallelization-in-rcpp-via-openmp/
// For Armadillo Support, check: http://arma.sourceforge.net/docs.html

// LDFLAGS:  -L/usr/local/opt/openblas/lib
// CPPFLAGS: -I/usr/local/opt/openblas/include

using namespace Rcpp;

int f(int i) {
  std::this_thread::sleep_for (std::chrono::seconds(1));
  return i;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]

int threadcheck(int i)
{
  int n_threads = i;
  std::cout << n_threads << " threads ..." << std::endl;
  std::vector<int> M(12);
#pragma omp parallel for num_threads(n_threads)
  for (int i=0; i<12; i++)
    M[i] = f(i);
  return 0;
}


Rcpp::List fixptfn_kushal(NumericVector pi_est,NumericMatrix matrix_lik, NumericVector prior){
//  clock_t start = clock();
  int n=matrix_lik.nrow(), k=matrix_lik.ncol();
  NumericVector pi_new(k);
  double loglik,lpriordens=0.0;
  NumericMatrix m(n,k);
  //  NumericMatrix classprob(m);
  NumericVector m_rowsum(n);
  //IntegerVector subset(prior);
//  clock_t end = clock();
//  double elapsed_time_0 = (end-start)/(double)CLOCKS_PER_SEC ;
//  std::cout << elapsed_time_0 << std::endl;
  
//  clock_t start1 = clock() ;
  for (int i=0;i<k;i++){
    pi_est[i]=std::max(0.0,pi_est[i]);
    m.column(i)=pi_est[i]*matrix_lik.column(i);
    m_rowsum=m_rowsum+m.column(i);
    if(prior[i]!=1.0){
      lpriordens +=(prior[i]-1.0)*log(pi_est[i]);
    }
  }
//  clock_t end1 = clock() ;
//  double elapsed_time_1 = (end1-start1)/(double)CLOCKS_PER_SEC ;
//  std::cout << elapsed_time_1 << std::endl;
  
  //  clock_t start2 = clock() ;
  //  #pragma omp parallel for num_threads(4)
  //  for (int i=0;i<k;i++){
  //    m.column(i)=m.column(i)/m_rowsum;
  //  }
  //  clock_t end2 = clock() ;
  //  double elapsed_time_2 = (end2-start2)/(double)CLOCKS_PER_SEC ;
  
 // clock_t start3 = clock() ;
  //calculating objective value--probability
  loglik=sum(log(m_rowsum));
//  clock_t end3 = clock() ;
//  double elapsed_time_3 = (end3-start3)/(double)CLOCKS_PER_SEC ;
//  std::cout << elapsed_time_3 << std::endl;
  
  //generating new pi
//  clock_t start4 = clock() ;
 #pragma omp parallel for num_threads(4)
  for (int i=0;i<k;i++){//set any estimates that are less than zero, which can happen with prior<1, to 0
    pi_new[i]=std::max(0.0,sum(m.column(i)/m_rowsum)+prior[i]-1.0);
  }
  
  pi_new=pi_new/sum(pi_new); //normalize pi
//  clock_t end4 = clock() ;
//  double elapsed_time_4 = (end4-start4)/(double)CLOCKS_PER_SEC ;
//  std::cout << elapsed_time_4 << std::endl;
  
  return(List::create(Named("fixedpointvector")=pi_new,
                      Named("objfn")=-loglik-lpriordens));
}



List squarem2(NumericVector par,NumericMatrix matrix_lik,NumericVector prior,List control){
  //control variables
  int maxiter=control["maxiter"];
  int method_=control["method"];
  bool trace=control["trace"];
  double stepmin0=control["step.min0"];
  double stepmax0=control["step.max0"];
  //double kr=control["kr"];
  double objfninc=control["objfn.inc"];
  double tol=control["tol"];
  double mstep=control["mstep"];
  
  List plist,p1list,p2list,p3list;
  NumericVector loldcpp,lnewcpp,pcpp,p1cpp,p2cpp,pnew,p;
  NumericVector q1,q2,sr2,sq2,sv2,srv;
  NumericVector lold,lnew;
  
  double sr2_scalar,sq2_scalar,sv2_scalar,srv_scalar,alpha,stepmin,stepmax;
  int iter,feval;
  bool conv,extrap;
  stepmin=stepmin0;
  stepmax=stepmax0;
  
  if(trace){Rcout<<"Squarem-1"<<std::endl;}
  iter=1;
  p=par;
  
  try{p1list=fixptfn_kushal(p,matrix_lik,prior);feval=1;}
  catch(...){
    Rcout<<"Error in fixptfn_kushal function evaluation";
    return 1;
  }
  p=p1list["fixedpointvector"];
  lold=p1list["objfn"];
  if(trace){Rcout<<"Objective fn: "<<lold[0]<<std::endl;}
  conv=true;
  
  p=par;
  
  while(feval<maxiter){
    //Step 1
    extrap = true;
    pcpp=p;
    try{p1list=fixptfn_kushal(p,matrix_lik,prior); feval++;}
    catch(...){
      Rcout<<"Error in fixptfn_kushal function evaluation";
      return 1;
    }
    p1cpp=p1list["fixedpointvector"];
    q1=p1cpp-pcpp;//
    sr2=q1*q1;
    sr2_scalar=0.0;
    for (int i=0;i<sr2.length();i++){sr2_scalar+=sr2[i];}
    if(sqrt(sr2_scalar)<tol){break;}
    
    
    //Step 2
    try{p2list=fixptfn_kushal(p1cpp,matrix_lik,prior);feval++;}
    catch(...){
      Rcout<<"Error in fixptfn_kushal function evaluation";
      return 1;
    }
    p2cpp=p2list["fixedpointvector"];
    lold=p2list["objfn"];
    q2=p2cpp-p1cpp;
    sq2=q2*q2;
    sq2_scalar=0;
    for (int i=0;i<q2.length();i++){sq2_scalar+=sq2[i];}
    sq2_scalar=sqrt(sq2_scalar);
    
    
    if (sq2_scalar<tol){break;}
    sv2=q2-q1;
    sv2_scalar=0;
    for (int i=0;i<sv2.length();i++){sv2_scalar+=sv2[i]*sv2[i];}
    srv_scalar=0;
    for (int i=0;i<q2.length();i++){srv_scalar+=sv2[i]*q1[i];}
    
    
    //Step 3 Proposing new value
    switch (method_){
    case 1: alpha= -srv_scalar/sv2_scalar;
    case 2: alpha= -sr2_scalar/srv_scalar;
    case 3: alpha= sqrt(sr2_scalar/sv2_scalar);
      //default: {
      //    Rcout<<"Misspecification in method, when K=1, method should be either 1, 2 or 3!";
      //    break;}
    }
    
    alpha=std::max(stepmin,std::min(stepmax,alpha));
    pnew = pcpp + 2.0*alpha*q1 + alpha*alpha*(q2-q1);
    
    //Step 4 stabilization
    if(std::abs(alpha-1)>0.01){
      try{p3list=fixptfn_kushal(pnew,matrix_lik,prior);feval++;}
      catch(...){
        pnew=p2cpp;
        lnew=p2list["objfn"];
        if(alpha==stepmax){
          stepmax=std::max(stepmax0,stepmax/mstep);
        }
        alpha=1;
        extrap=false;
        if(alpha==stepmax){stepmax=mstep*stepmax;}
        if(stepmin<0.0 && alpha==stepmin){stepmin=mstep*stepmin;}
        p=pnew;
        lnewcpp=lnew;
        if(!std::isnan(lnewcpp[0])){lold=lnew;}
        if(trace){Rcout<<"Objective fn: "<<lnewcpp[0]<<"  Extrapolation: "<<extrap<<"  Steplength: "<<alpha<<std::endl;}
        iter++;
        continue;//next round in while loop
      }
      pnew=p3list["fixedpointvector"];
      lnew=p3list["objfn"];
      lnewcpp=lnew;
      if (lnewcpp[0]>loldcpp[0]+objfninc) {
        pnew=p2list["fixedpointvector"];
        lnew=p2list["objfn"];
        if(alpha==stepmax){
          stepmax=std::max(stepmax0,stepmax/mstep);
        }
        alpha=1;
        extrap=false;
      }
    }else{//same as above, when stablization is not performed.
      lnew=lold;
      lnewcpp=lnew;
      if (lnewcpp[0]>loldcpp[0]+objfninc) {
        pnew=p2list["fixedpointvector"];
        lnew=p2list["objfn"];
        if(alpha==stepmax){
          stepmax=std::max(stepmax0,stepmax/mstep);
        }
        alpha=1;
        extrap=false;
      }
    }
    
    if(alpha==stepmax){stepmax=mstep*stepmax;}
    if(stepmin<0 && alpha==stepmin){stepmin=mstep*stepmin;}
    
    p=pnew;
    lnewcpp=lnew;
    if(!std::isnan(lnewcpp[0])){lold=lnew;}
    loldcpp=lold;
    if(trace){Rcout<<"Objective fn: "<<lnewcpp[0]<<"  Extrapolation: "<<extrap<<"  Steplength: "<<alpha<<std::endl;}
    
    iter++;
  }
  
  if (feval >= maxiter){conv=false;}
  
  return(List::create(Named("par")=p,
                      Named("value.objfn")=lold,
                      Named("iter")=iter,
                      Named("fpevals")=feval,
                      Named("objfevals")=feval,
                      Named("convergence")=conv));
}



// [[Rcpp::export]]
List cxxMixSquarem(NumericMatrix matrix_lik, NumericVector prior, NumericVector pi_init, List control){//note: no default pi_init=NULL
  int  k=matrix_lik.ncol(),niter;
  bool converged=NA_LOGICAL;
  double loglik;
  List res;
  NumericVector pi(k);
  
  if(Rf_isNull(pi_init))
    std::fill(pi.begin(), pi.end(), 1./(double)k);
  else{
    pi=clone(pi_init);
  }
  
  res=squarem2(pi,matrix_lik,prior,control);
  pi=res["par"];
  loglik=res["value.objfn"];
  niter=res["iter"];
  converged=res["convergence"];
  pi=pi/sum(pi); //normalize pi again
  return(List::create(Named("pihat")=pi,
                      Named("B")=loglik,
                      Named("niter")=niter,
                      Named("converged")=wrap(converged)));
}


/***R
prior <- rep(1,10000);
pi_est <- gtools::rdirichlet(1, rchisq(10000,4));
matrix_lik <- matrix(rchisq(1000000,2), ncol=10000);
control.default=list(K = 1, method=3, square=TRUE, 
                     step.min0=1, step.max0=1, mstep=4, kr=1, 
                     objfn.inc=1,tol=1.e-07, maxiter=10000, trace=FALSE)
system.time(for(m in 1:1){
  ll4 <- cxxMixSquarem2(matrix_lik, prior, pi_init=pi_est, control=control.default)
})

*/
