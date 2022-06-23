# mediation_drf_gps
The code accompanying Alexander Gudjonsson's master thesis from ETH Zürich, Seminar for Statistics, titled "Non-parametric Causal Mediation Analysis with Distributional Random Forests and Generalised Propensity Scores"

# mediation_drf_gps.py / mediation_drf_covadj.py
The causal mediation estimators using distributional random forest and generalised propensity scores / covariate adjustment as described in the thesis.

# evaluation.py
A class built around the simulation experiments described in the thesis.
It takes in a data generating model and does the following:
 - Simulates multiple datasets from it.
 - Finds the true mediating, direct and total effect curves.
 - Trains the mediation estimator on each of the datasets.
 - Predicts the three effect curves on the nine deciles of the treatment variable.
 - Aggregates the results and calculates summary statistics.
 - Returns two files, one with the mean estimated function values and 95% confidence intervals,
   the other with overall mean-squared-error, bias and variance.


# model1.py / model2.py / model3.py
The data generating models corresponding to Model 1/2/3 in the thesis. As it is, the scripts produce results for the covariate adjustment estimator with normally distributed errors and the propensity score estimator with normally and t distributed errors. The sample size can be changed to obtain the results in the benchmark table corresponding to the linear model (Model 1) and non-linear model (Model 3) with p=4.

# model4.py / model5.py
The data generating model corresponding to Model 4/5 in the thesis. The script produces results with base treatment at the 3rd and 7th treatment deciles.

# model_benchmark_linear.py / model_benchmark_nonlinear.py
Same data generating models as Model 1 and 3, respectfully, but with a tunable dimension of X, used for the benchmark table 

# model_conf_TM.py / model_conf_TY.py / model_conf_MY.py / model_conf_postT.py
The data generating models corresponding to Model 3 in the thesis, with a hidden confounder affecting variables T-M, T-Y, M-Y and posttreatment M-Y, respectfully. The standard deviation of the hidden confounders, h, can be tuned to obtain the results in table 4.2.

# model1_Huber.R / model3_Huber.R / model_benchmark_linear_Huber.R / model_benchmark_nonlinear_Huber.R
Corresponding models run with the estimator proposed by Huber et al. to produce results present in table 4.1 in the thesis.

# evaluation_Huber.R
The estimator proposed by Huber et al. set up in a similar framework as the evaluation.py to produce results present in table 4.1 in the thesis.
