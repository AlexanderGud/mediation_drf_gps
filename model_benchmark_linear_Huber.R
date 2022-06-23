# Same data generating model as Model 1 but with a tunable dimension of X,
# used to produce results present in table 4.1 in the thesis.

source('evaluation_Huber.R')

n = 100 # sample size
p = 20 # dimension of X, tunable

gen_error_norm = function(n,p){
  matrix(rnorm(n*(p+5)), nrow = n)
}
gen_error_t = function(n,p){
  matrix(rt(n*(p+5),3), nrow = n)
}
gen_X = function(e){
  e[,6:ncol(e)]
}
gen_H = function(e,X){ # unused hidden confounder
  e[,1]
}
gen_Tr = function(e,X,H=0){
  q = ncol(X)/4
  coeff = 1/sqrt(q)*seq(0.5,1.5,length.out=3*q)
  covar = X[,1:(3*q)]
  covar%*%coeff + e[,2]
}
gen_Hpost = function(e,Tr){ # unused hidden confounder
  e[,3]
}
gen_M = function(e,X,Tr,H=0,Hpost=0){
  q = ncol(X)/4
  coeff = 1/sqrt(q)*seq(1,3,length.out=3*q)*rep_len(c(1,-1),3*q)
  covar = X[,c(1:(2*q),(3*q+1):(4*q))]
  0.5*Tr + covar%*%coeff + e[,4]
}
gen_Y = function(e,X,Tr,M,H=0,Hpost=0){
  q = ncol(X)/4
  coeff = 1/sqrt(q)*seq(2,1,length.out=3*q)*rep_len(c(1,1-1,-1),3*q)
  covar = X[,c(1:q,(2*q+1):(4*q))]
  3*M + 1*Tr + covar%*%coeff + e[,5]
}

gen_e = gen_error_norm
modelname = paste0('model_benchmark_linear_Huber_enorm_p', p, '_n', n) 
mediation_evaluation_Huber()

gen_e = gen_error_t
modelname = paste0('model_benchmark_linear_Huber_et_p', p, '_n', n) 
mediation_evaluation_Huber()
