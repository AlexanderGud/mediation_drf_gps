# A class built around the simulation experiments described in the thesis.
#  It takes in a data generating model and does the following:
#  - Simulates multiple datasets from it.
#  - Finds the true mediating, direct and total effect curves.
#  - Trains the mediation estimator on each of the datasets.
#  - Predicts the three effect curves on the nine deciles of the treatment variable.
#  - Aggregates the results and calculates summary statistics.
#  - Returns two files, one with the mean estimated function values and 95% confidence intervals,
#    the other with overall mean-squared-error, bias and variance.

# Inputs
# ------
# model: a dictionary of data generating function, generating each variable in the model
# n: number of samples
# p: dimension of X
# modelname: name for the output files
# base: the decile number base of the treatment variable will be used as the base treatment value
#       base = 5 means that the median (5th decile) will be used as the base treatment.
# method: "gps" or "covadj" for the estimator using generalised propensity scores or covariate adjustment
# num_sim: number of simulations
# n_jobs: number of parallel jobs
# seed: seed for reproducability

import numpy as np
import pandas as pd
from multiprocessing import Pool
from drf import drf
from mediation_drf_gps import causal_mediation as causal_mediation_gps
from mediation_drf_covadj import causal_mediation as causal_medaition_covadj
from sklearn.neural_network import MLPRegressor as nn
from sklearn.model_selection import GridSearchCV


class mediation_evaluation:
    def __init__(
        self,
        model,
        n,
        p,
        modelname,
        base=5,
        method="gps",
        num_sim=100,
        n_jobs=1,
        seed=None,
    ):
        self.gen_e = model["gen_e"]
        self.gen_X = model["gen_X"]
        self.gen_H = model["gen_H"]
        self.gen_T = model["gen_T"]
        self.gen_Hpost = model["gen_Hpost"]
        self.gen_M = model["gen_M"]
        self.gen_Y = model["gen_Y"]
        self.n = n
        self.p = p
        self.modelname = modelname
        self.base = base
        if method is "gps":
            self.causal_mediation = causal_mediation_gps
        elif method is "covadj":
            self.causal_mediation = causal_mediation_covadj
        self.num_sim = num_sim
        self.n_jobs = n_jobs
        self.seed = seed
        np.random.seed(seed)
        if seed is None:
            self.seedR = np.random.randint(
                1e6
            )  # the drf calls R, which cannot interpret they keyword "None"
        else:
            self.seedR = seed

        self.find_true_effects()
        self.define_model()
        self.estimate_effects()

    def gen_data(self):
        # generate an instance of the data
        e = self.gen_e(self.n, self.p)
        X = self.gen_X(e)
        H = self.gen_H(e, X)
        T = self.gen_T(e, X, H)
        Hpost = self.gen_Hpost(e, T)
        M = self.gen_M(e, X, T, H, Hpost)
        Y = self.gen_Y(e, X, T, M, H, Hpost)
        return {"X": X, "T": T, "M": M, "Y": Y}

    def find_true_effects(self):
        # calculate the true effect curves at the nine deciles of the treatment.
        # Done with the data generating functions but with a large sample size
        # and no errors for post-treatment variables.
        n_ = int(1e6)
        e_ = self.gen_e(n_, self.p)
        e_no = np.zeros((n_, e_.shape[1]-self.p))                                                    
        X_ = self.gen_X(e_)
        H_ = self.gen_H(e_, X_)
        T_ = self.gen_T(e_, X_, H_)
        self.treatvalues = np.quantile(T_, np.linspace(0.1, 0.9, 9))
        self.l = len(self.treatvalues)
        T0 = np.full(n_, self.treatvalues[self.base - 1])
        Hpost0 = self.gen_Hpost(e_no, T0)
        M0 = self.gen_M(e_no, X_, T0, H_, Hpost0)
        med_true = np.zeros(self.l)
        dir_true = np.zeros(self.l)
        total_true = np.zeros(self.l)
        for i in range(self.l):
            T1 = np.full(n_, self.treatvalues[i])
            Hpost1 = self.gen_Hpost(e_no, T1)
            M1 = self.gen_M(e_no, X_, T1, H_, Hpost1)
            Ymed = self.gen_Y(e_no, X_, T0, M1, H_, Hpost0)
            Ydir = self.gen_Y(e_no, X_, T1, M0, H_, Hpost1)
            Ytot = self.gen_Y(e_no, X_, T1, M1, H_, Hpost1)
            med_true[i] = Ymed.mean()
            dir_true[i] = Ydir.mean()
            total_true[i] = Ytot.mean()

        self.true_val = np.hstack(
            [
                med_true.reshape((self.l, 1)),
                dir_true.reshape((self.l, 1)),
                total_true.reshape((self.l, 1)),
            ]
        )

    def define_model(self):
        # Define the parameters of the models and initialise
        drf_M = drf(num_trees=500, seed=self.seedR)
        nn_params = {"alpha": 10.0 ** -np.arange(2, 5)}
        fit_Y = GridSearchCV(
            nn(max_iter=2000, hidden_layer_sizes=(2 * self.p, self.p)),
            nn_params,
            n_jobs=1,
        )
        self.medmodel = self.causal_mediation(
            drf_M, fit_Y, k_M=1000, k_X=1000, seed=self.seed
        )

    def estimate_effects_per_sim(self, seed):
        # Simulate one batch of data, fit the models and predict the effects
        np.random.seed(seed)
        data = self.gen_data()

        self.medmodel.fit(data)

        med_est = np.zeros(self.l)
        dir_est = np.zeros(self.l)
        total_est = np.zeros(self.l)

        for i in range(self.l):
            med_est[i] = self.medmodel.predict(
                t_d=self.treatvalues[self.base - 1], t_m=self.treatvalues[i]
            )
            dir_est[i] = self.medmodel.predict(
                t_d=self.treatvalues[i], t_m=self.treatvalues[self.base - 1]
            )
            total_est[i] = self.medmodel.predict(
                t_d=self.treatvalues[i], t_m=self.treatvalues[i]
            )

        return np.hstack([med_est, dir_est, total_est])

    def estimate_effects(self):
        # Call the function estimate_effects_per_sim repeatedly in parallel
        # Aggregate results and produce csv files

        with Pool(self.n_jobs) as p:
            est = np.array(p.map(self.estimate_effects_per_sim, range(self.num_sim)))

        est_mean = est.mean(axis=0).reshape((self.l, 3), order="F")
        est_95lb = np.quantile(est, axis=0, q=0.025).reshape((self.l, 3), order="F")
        est_95ub = np.quantile(est, axis=0, q=0.975).reshape((self.l, 3), order="F")
        est_var = est.var(axis=0).reshape((self.l, 3), order="F")
        est_bias2 = (est_mean - self.true_val) ** 2
        est_mse = est_bias2 + est_var

        res = pd.DataFrame(
            np.hstack(
                [
                    self.treatvalues[0 : self.l].reshape((self.l, 1)),
                    self.true_val,
                    est_mean,
                    est_95lb,
                    est_95ub,
                ]
            ),
            columns=[
                "treatment",
                "med_true",
                "dir_true",
                "total_true",
                "med_mean",
                "dir_mean",
                "total_mean",
                "med_95lb",
                "dir_95lb",
                "total_95lb",
                "med_95ub",
                "dir_95ub",
                "total_95ub",
            ],
        )

        y_range = self.true_val.max() - self.true_val.min()
        mse = (
            pd.DataFrame(
                np.hstack([est_bias2, est_var, est_mse])
                .mean(axis=0)
                .reshape((3, 3), order="F")
            )
            / y_range
        )
        mse.columns = ["bias^2", "var", "mse"]
        mse.index = ["med", "dir", "total"]

        res.to_csv(
            self.modelname + "_est.csv",
            index=False,
        )
        mse.to_csv(
            self.modelname + "_mse.csv",
            index=True,
        )
