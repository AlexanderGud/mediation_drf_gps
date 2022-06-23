# The data generating model corresponding to Model 3 in the thesis,
# with a hidden confounder affecting T and Y.

from evaluation import mediation_evaluation
import numpy as np

n = 5000  # sample size
p = 4  # dimension of X, hardcoded at 4 in this model
h = 1  # standard deviation of the confounding effect
n_jobs = 1  # number of jobs for parallelisation


def gen_error_norm(n, p):
    return np.random.normal(0, 1, (n, (p + 5)))


def gen_X(e):
    return e[:, 5:]


def gen_H(e, X):
    return h * e[:, 0]


def gen_T(e, X, H=0):
    return H + X[:, 0] + X[:, 1] + X[:, 2] + e[:, 1]


def gen_Hpost(e, T):  # unused hidden confounder
    return e[:, 2]


def gen_M(e, X, T, H=0, Hpost=0):
    return (2 + np.cos(T) + X[:, 0] - X[:, 1] + 2 * X[:, 3] + e[:, 3]).reshape(
        (len(X), 1)
    )


def gen_Y(e, X, T, M, H=0, Hpost=0):
    return (
        3 * np.sqrt(abs(M[:, 0]))
        - np.arctan(T)
        + H
        - 0.5 * X[:, 0]
        - 2 * X[:, 2]
        - X[:, 3]
        + e[:, 4]
    )


model = {
    "gen_e": gen_error_norm,
    "gen_X": gen_X,
    "gen_H": gen_H,
    "gen_T": gen_T,
    "gen_Hpost": gen_Hpost,
    "gen_M": gen_M,
    "gen_Y": gen_Y,
}

modelname = "model_conf_TY_h" + str(h) + "_enorm_p" + str(p) + "_n" + str(n)
mediation_evaluation(model, n, p, modelname, n_jobs=n_jobs, seed=30)
