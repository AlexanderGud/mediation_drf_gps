# The estimator proposed by Huber et al. set up in a similar framework as
# the evaluation.py to produce results present in table 4.1 in the thesis.

library(causalweight)

gen_data = function(n,p){
  e = gen_e(n,p)
  X = gen_X(e)
  H = gen_H(e,X)
  Tr = gen_Tr(e,X,H)
  Hpost = gen_Hpost(e,Tr)
  M = gen_M(e,X,Tr,H,Hpost)
  Y = gen_Y(e,X,Tr,M,H,Hpost)
  return(list(X=X,Tr=Tr,M=M,Y=Y))
}
find_true_effects = function(){
  n_ = as.integer(1e6)
  e_ = gen_e(n_,p)
  e_no = matrix(rep(0,n_*5), nrow = n_)
  X_ = gen_X(e_)
  H_ = gen_H(e_,X_)
  Tr_ = gen_Tr(e_,X_,H_)
  treatvalues = quantile(Tr_, seq(0.1,0.9,0.1))
  l = length(treatvalues)
  T0 = rep(treatvalues[4], n_)
  Hpost0 = gen_Hpost(e_no,T0)
  M0 = gen_M(e_no,X_,T0,H_,Hpost0)
  med_true = rep(0,l)
  dir_true = rep(0,l)
  total_true = rep(0,l)
  for( i in 1:l ){
    T1 = rep(treatvalues[i], n_)
    Hpost1 = gen_Hpost(e_no,T1)
    M1 = gen_M(e_no,X_,T1,H_,Hpost1)
    Ymed = gen_Y(e_no,X_,T0,M1,H_,Hpost0)
    Ydir = gen_Y(e_no,X_,T1,M0,H_,Hpost1)
    Ytot = gen_Y(e_no,X_,T1,M1,H_,Hpost1)
    med_true[i] = mean(Ymed)
    dir_true[i] = mean(Ydir)
    total_true[i] = mean(Ytot)
  }
  return(cbind(treatvalues, total_true, dir_true, med_true))
}
predict_eff = function(data,t,base){
  res = medweightcont(data$Y,data$Tr,data$M,data$X,d0=base,d1=t,trim=Inf)
  return(res$results['effect',c('ATE', 'dir.control', 'indir.control')])
}
est_per_sim = function(treatvalues,base){
  data = gen_data(n,p)
  return(sapply(treatvalues, function(t) predict_eff(data,t,base)))
}
mediation_evaluation_Huber = function(){
  set.seed(30)
  truth = find_true_effects()
  truth_flat = c(t(truth[,-1]))
  treatvalues = truth[,'treatvalues']
  l = length(treatvalues)
  base = treatvalues[5]
  base_response = truth[5,'total_true']
  boot_est = sapply(1:100, function(z) est_per_sim(treatvalues,base)) + base_response
  est_mean = matrix(rowMeans(boot_est, na.rm = TRUE), nrow = l, byrow = TRUE)
  est_95lb = matrix(apply(boot_est, 1, function(x) quantile(x, probs = 0.025, names = FALSE, na.rm = TRUE)), nrow = l, byrow = TRUE)
  est_95ub = matrix(apply(boot_est, 1, function(x) quantile(x, probs = 0.975, names = FALSE, na.rm = TRUE)), nrow = l, byrow = TRUE)
  est = data.frame(cbind(truth, est_mean, est_95lb, est_95ub))
  colnames(est) = c('treatment', 'total_true', 'dir_true', 'med_true', 'total_mean', 'dir_mean', 'med_mean', 'total_95lb', 'dir_95lb', 'med_95lb', 'total_95ub', 'dir_95ub', 'med_95ub')
  mse = colMeans(matrix(rowMeans((boot_est - truth_flat)**2, na.rm=TRUE), nrow = l, byrow = TRUE))
  y_range = max(truth[,-1]) - min(truth[,-1])
  mse_norm = mse / y_range
  names(mse_norm) = c('total', 'dir', 'med')
  
  write.table(est, paste0(modelname, '_est.csv'), row.names = FALSE, col.names = TRUE, quote = FALSE)
  write.table(t(mse_norm), paste0(modelname, '_mse.csv'), row.names = FALSE, col.names = TRUE, quote = FALSE)
}