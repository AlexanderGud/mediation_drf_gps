# The causal mediation estimator using distributional random forest and covariate adjustment
# as described in the thesis.

# Inputs
# ------
# fit_M: an initialised but unfitted distributional random forest model
# fit_Y: an initialised but unfitted regressor with a predict attribute (e.g. from sklearn)
# k_X: the Monte Carlo sampling size of the outer integral, integrating over X
# k_M: the Monte Carlo sampling size of the inner integral, integrating over M
# seed: for reproducability

import numpy as np


class causal_mediation:
    def __init__(self, fit_M, fit_Y, k_X=1000, k_M=1000, seed=None):
        self.fit_M = fit_M
        self.fit_Y = fit_Y
        self.k_X = k_X
        self.k_M = k_M
        np.random.seed(seed)

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

        feat_XT = np.hstack([self.X, self.T])
        feat_XTM = np.hstack([feat_XT, self.M])

        self.p_XTM = feat_XTM.shape[1]

        self.fit_M.fit(feat_XT, self.M)
        self.fit_Y.fit(feat_XTM, self.Y)

    def predict(self, t_d, t_m):
        # estimate integral{ integral( E[Y|T=t_d,X=x,M=m] )dF(m|T=t_m,X=x) }dx
        X_sample = self.X[np.random.choice(range(self.n), self.k_X), :]
        M_cond_weights = self.fit_M.predict(
            np.hstack([X_sample, np.full((self.k_X, 1), t_m)])
        ).weights
        pred_data = np.zeros((self.k_X * self.k_M, self.p_XTM))
        for i in range(self.k_X):
            M_sample = self.M[
                np.random.choice(np.arange(self.n), self.k_M, p=M_cond_weights[i, :]), :
            ]
            pred_data[i * self.k_M : (i + 1) * self.k_M, :] = np.hstack(
                [
                    np.full((self.k_M, len(X_sample[i, :])), X_sample[i, :]),
                    np.full((self.k_M, 1), t_d),
                    M_sample,
                ]
            )
        Y_pred = self.fit_Y.predict(pred_data).mean()
        return Y_pred
