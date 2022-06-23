# Model 1 in the thesis run with the estimator proposed by Huber et al.
# to produce results present in table 4.1 in the thesis.

source('evaluation_Huber.R')

n = 100 # sample size
p = 4 # dimension of X, hardcoded at 4 in this model

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
  X[,1] + X[,2] + X[,3] + e[,2]
}
gen_Hpost = function(e,Tr){ # unused hidden confounder
  e[,3]
}
gen_M = function(e,X,Tr,H=0,Hpost=0){
  0.5*Tr + X[,1] - X[,2] + 2*X[,4] + e[,4]
}
gen_Y = function(e,X,Tr,M,H=0,Hpost=0){
  3*M + Tr - 0.5*X[,1] - 2*X[,3] - X[,4] + e[,4]
}

gen_e = gen_error_norm
modelname = paste0('model1_Huber_enorm_p', p, '_n', n) 
mediation_evaluation_Huber()

gen_e = gen_error_t
modelname = paste0('model1_Huber_et_p', p, '_n', n) 
mediation_evaluation_Huber()
