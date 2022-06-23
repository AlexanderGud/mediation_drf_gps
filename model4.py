# The data generating model corresponding to Model 4 in the thesis.

from evaluation import mediation_evaluation
import numpy as np

n = 5000  # sample size
p = 4  # dimension of X, hardcoded at 4 in this model
n_jobs = 1  # number of jobs for parallelisation


def gen_error_norm(n, p):
    return np.random.normal(0, 1, (n, (p + 5)))


def gen_X(e):
    return e[:, 5:]


def gen_H(e, X):  # unused hidden confounder
    return e[:, 0]


def gen_T(e, X, H=0):
    return 2 + X[:, 0] + X[:, 1] + X[:, 2] + e[:, 1]


def gen_Hpost(e, T):  # unused hidden confounder
    return e[:, 2]


def gen_M(e, X, T, H=0, Hpost=0):
    return (T - X[:, 0] + 0.5 * X[:, 1] + 1.5 * X[:, 3] + e[:, 3]).reshape((len(X), 1))


def gen_Y(e, X, T, M, H=0, Hpost=0):
    return M[:, 0] * np.arctan(T) - X[:, 0] + 0.5 * X[:, 2] - 0.5 * X[:, 3] + e[:, 4]


model = {
    "gen_e": gen_error_norm,
    "gen_X": gen_X,
    "gen_H": gen_H,
    "gen_T": gen_T,
    "gen_Hpost": gen_Hpost,
    "gen_M": gen_M,
    "gen_Y": gen_Y,
}

modelname = "model4_base3_enorm_p" + str(p) + "_n" + str(n)
mediation_evaluation(model, n, p, modelname, base=3, n_jobs=n_jobs, seed=30)

modelname = "model4_base7_enorm_p" + str(p) + "_n" + str(n)
mediation_evaluation(model, n, p, modelname, base=7, n_jobs=n_jobs, seed=30)
