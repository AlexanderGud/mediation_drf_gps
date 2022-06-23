# The causal mediation estimator using distributional random forest and generalised propensity scores
# as described in the thesis.

# Inputs
# ------
# fit_M: an initialised but unfitted distributional random forest model
# fit_Y: an initialised but unfitted regressor with a predict attribute (e.g. from sklearn)
# k_X: the Monte Carlo sampling size of the outer integral, integrating over X
# k_M: the Monte Carlo sampling size of the inner integral, integrating over M
# kernelbandwidth: the kernel bandwidth of the density estimator for the generalised propensity function
# seed: for reproducability

import numpy as np
from drf import drf
from sklearn.neighbors import KernelDensity


class causal_mediation:
    def __init__(self, fit_M, fit_Y, k_X=1000, k_M=1000, kernelbandwidth=1, seed=None):
        self.fit_M = fit_M
        self.fit_Y = fit_Y
        self.k_X = k_X
        self.k_M = k_M
        self.kernelbandwidth = kernelbandwidth
        self.seed = seed
        np.random.seed(seed)
        if seed is None:
            self.seedR = np.random.randint(
                1e6
            )  # the drf calls R, which cannot interpret the keyword None
        else:
            self.seedR = seed

    def fit(self, data):
        self.X = data["X"]
        self.T = data["T"]
        self.M = data["M"]
        self.Y = data["Y"]
        self.n = len(self.X)

        if len(self.X.shape) == 1:
            self.X = self.X.reshape((self.n, 1))
        if len(self.T.shape) == 1:
            self.T = self.T.reshape((self.n, 1))
        if len(self.M.shape) == 1:
            self.M = self.M.reshape((self.n, 1))

        self.GPF_T = self.GPF(self.X, self.T, self.kernelbandwidth)
        GPS_T = self.GPS(self.GPF_T, self.T)

        feat_DRF = np.hstack([self.T, GPS_T])
        feat_reg = np.hstack([self.M, self.T, self.X, GPS_T])

        self.p_reg = feat_reg.shape[1]

        self.fit_M.fit(feat_DRF, self.M)
        self.fit_Y.fit(feat_reg, self.Y)

    def GPF(self, X, Y, kernelbandwidth):
        # estimate the Generalised Propensity Function for each sample
        l = len(Y)
        DRF_GPF = drf(seed=self.seedR)
        DRF_GPF.fit(X, Y)
        w = DRF_GPF.predict(X).weights
        GPF = list()
        for i in range(l):
            Y_cond_sample = Y[np.random.choice(np.arange(l), l, p=w[i, :]), :]
            cond_density = KernelDensity(bandwidth=kernelbandwidth).fit(Y_cond_sample)
            GPF.append(cond_density)
        return np.array(GPF)

    def GPS(self, GPF, x):
        # estimate the Generalised Propensity Score by evaluating the GPF at a certain point
        # assumes GPF is a list of KernelDensity and x is an array of the same length
        l = len(GPF)
        GPS = np.zeros((l, 1))
        for i in range(l):
            GPS[i] = np.exp(GPF[i].score_samples([x[i]]))
        return GPS

    def predict(self, t_d, t_m):
        # estimate integral{ integral( E[Y|M=m,T=t_d,X,r(t_d,x)] )dF(m|T=t_m,r(t_m,x)) }dx
        sample_ind = np.random.choice(range(self.n), self.k_X)
        X_sample = self.X[sample_ind, :]
        GPS_Tm_sample = self.GPS(self.GPF_T[sample_ind], np.full((self.k_X, 1), t_m))
        GPS_Td_sample = self.GPS(self.GPF_T[sample_ind], np.full((self.k_X, 1), t_d))
        Tm = np.full((self.k_X, 1), t_m)
        M_cond_weights = self.fit_M.predict(np.hstack([Tm, GPS_Tm_sample])).weights
        pred_data = np.zeros((self.k_X * self.k_M, self.p_reg))
        for i in range(self.k_X):
            M_sample = self.M[
                np.random.choice(np.arange(self.n), self.k_M, p=M_cond_weights[i, :]), :
            ]
            Td = np.full((self.k_M, 1), t_d)
            X_sample_i = np.full((self.k_M, len(X_sample[i, :])), X_sample[i, :])
            GPS_Td_sample_i = np.full((self.k_M, 1), GPS_Td_sample[i])
            pred_data[i * self.k_M : (i + 1) * self.k_M, :] = np.hstack(
                [M_sample, Td, X_sample_i, GPS_Td_sample_i]
            )
        Y_pred = self.fit_Y.predict(pred_data).mean()
        return Y_pred
