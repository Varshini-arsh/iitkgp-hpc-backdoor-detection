from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

COUNTERS = ["cycles", "cache-references", "LLC-loads"]
FEATURES = [f"log_{c.replace('-', '_')}" for c in COUNTERS]


def load_hpc(path):
    frame = pd.read_csv(path)
    for col in list(frame.columns):
        if str(col).startswith("Unnamed:"):
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy()
            if np.isfinite(values).all() and np.array_equal(values, np.arange(len(frame))):
                frame = frame.drop(columns=[col])
    missing = [c for c in COUNTERS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required HPC columns: {missing}; found {list(frame.columns)}")
    raw = frame[COUNTERS].apply(pd.to_numeric, errors="coerce")
    values = raw.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("HPC columns contain missing or non-finite values.")
    if (values <= 0).any():
        raise ValueError("All selected HPC counters must be positive.")
    return frame, raw


def transform(raw):
    return np.log(raw[COUNTERS].to_numpy(dtype=float))


def fit_gmm(features, seed=42, max_components=5):
    scaler = StandardScaler().fit(features)
    z = scaler.transform(features)
    upper = min(max_components, max(1, len(z) // 20))
    candidates = []
    for k in range(1, upper + 1):
        model = GaussianMixture(
            n_components=k, covariance_type="full", reg_covar=1e-5,
            n_init=5, random_state=seed,
        ).fit(z)
        candidates.append((model.bic(z), model))
    bic, model = min(candidates, key=lambda pair: pair[0])
    return scaler, model, float(bic)


def conditional_nll(z, model):
    """Per-counter -log p(x_j | x_-j) under a fitted full-covariance GMM."""
    z = np.asarray(z, dtype=float)
    n, d = z.shape
    component_count = model.n_components
    result = np.zeros((n, d), dtype=float)
    log_weights = np.log(np.maximum(model.weights_, 1e-300))
    log2pi = np.log(2.0 * np.pi)

    for j in range(d):
        other = np.array([i for i in range(d) if i != j])
        component_log_joint = np.empty((n, component_count), dtype=float)
        component_log_conditional = np.empty((n, component_count), dtype=float)

        for k in range(component_count):
            mu = model.means_[k]
            cov = model.covariances_[k]
            cov_oo = cov[np.ix_(other, other)]
            cov_oj = cov[other, j]
            sign, logdet = np.linalg.slogdet(cov_oo)
            if sign <= 0:
                raise ValueError("Non-positive covariance encountered in conditional density.")
            delta = z[:, other] - mu[other]
            solved = np.linalg.solve(cov_oo, delta.T).T
            log_p_other = -0.5 * (len(other) * log2pi + logdet + np.sum(delta * solved, axis=1))
            beta = np.linalg.solve(cov_oo, cov_oj)
            cond_mean = mu[j] + delta @ beta
            cond_var = max(float(cov[j, j] - cov[j, other] @ beta), 1e-10)
            log_p_j_given_other = -0.5 * (
                log2pi + np.log(cond_var) + (z[:, j] - cond_mean) ** 2 / cond_var
            )
            component_log_joint[:, k] = log_weights[k] + log_p_other
            component_log_conditional[:, k] = log_p_j_given_other

        log_marginal_other = logsumexp(component_log_joint, axis=1)
        log_joint = logsumexp(component_log_joint + component_log_conditional, axis=1)
        result[:, j] = -(log_joint - log_marginal_other)
    return result


def calibrated_threshold(calibration_scores, alpha):
    return float(np.quantile(np.asarray(calibration_scores), 1.0 - alpha, method="higher"))
