# Same data generating model as Model 3 but with a tunable dimension of X,
# used to produce results present in table 4.1 in the thesis.

from evaluation import mediation_evaluation
import numpy as np

n = 100  # sample size
p = 20  # dimension of X, tunable
n_jobs = 1  # number of jobs for parallelisation


def gen_error_norm(n, p):
    return np.random.normal(0, 1, (n, (p + 5)))


def gen_error_t(n, p):
    return np.random.standard_t(3, (n, (p + 5)))


def gen_X(e):
    return e[:, 5:]


def gen_H(e, X):  # unused hidden confounder
    return e[:, 0]


def gen_T(e, X, H=0):
    q = int(X.shape[1] / 4)
    coeff = 1 / np.sqrt(q) * np.linspace(0.5, 1.5, 3 * q) * np.resize([1], 3 * q)
    covar = X[:, np.arange(3 * q)]
    return (covar * coeff).sum(axis=1) + e[:, 1]


def gen_Hpost(e, T):  # unused hidden confounder
    return e[:, 2]


def gen_M(e, X, T, H=0, Hpost=0):
    q = int(X.shape[1] / 4)
    coeff = 1 / np.sqrt(q) * np.linspace(1, 3, 3 * q) * np.resize([1, -1], 3 * q)
    covar = X[:, np.hstack([np.arange(2 * q), np.arange(3 * q, 4 * q)])]
    return (2 + np.cos(T) + (covar * coeff).sum(axis=1) + e[:, 3]).reshape((len(e), 1))


def gen_Y(e, X, T, M, H=0, Hpost=0):
    q = int(X.shape[1] / 4)
    coeff = 1 / np.sqrt(q) * np.linspace(2, 1, 3 * q) * np.resize([1, 1, -1, -1], 3 * q)
    covar = X[:, np.hstack([np.arange(q), np.arange(2 * q, 4 * q)])]
    return (
        3 * np.sqrt(np.abs(M[:, 0]))
        - np.arctan(T)
        + (covar * coeff).sum(axis=1)
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

modelname = "model_benchmark_nonlinear_enorm_p" + str(p) + "_n" + str(n)
mediation_evaluation(model, n, p, modelname, n_jobs=n_jobs, seed=30)

model = {
    "gen_e": gen_error_t,
    "gen_X": gen_X,
    "gen_H": gen_H,
    "gen_T": gen_T,
    "gen_Hpost": gen_Hpost,
    "gen_M": gen_M,
    "gen_Y": gen_Y,
}

modelname = "model_benchmark_nonlinear_et_p" + str(p) + "_n" + str(n)
mediation_evaluation(model, n, p, modelname, n_jobs=n_jobs, seed=30)
