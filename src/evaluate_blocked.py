import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detector_core import fit_gmm, load_hpc, transform, conditional_nll


def score_joint(train, cal, test, seed, max_components):
    scaler, model, _ = fit_gmm(train, seed, max_components)
    c, t = scaler.transform(cal), scaler.transform(test)
    return -model.score_samples(c), -model.score_samples(t)


def score_conditional(train, cal, test, seed, max_components):
    scaler, model, _ = fit_gmm(train, seed, max_components)
    c, t = scaler.transform(cal), scaler.transform(test)
    return conditional_nll(c, model).mean(axis=1), conditional_nll(t, model).mean(axis=1)


def score_independent(train, cal, test, seed, max_components):
    cparts, tparts = [], []
    for j in range(train.shape[1]):
        scaler, model, _ = fit_gmm(train[:, [j]], seed + j, max_components)
        c, t = -model.score_samples(scaler.transform(cal[:, [j]])), -model.score_samples(scaler.transform(test[:, [j]]))
        mu, sd = c.mean(), max(c.std(), 1e-8)
        cparts.append((c - mu) / sd)
        tparts.append((t - mu) / sd)
    return np.max(np.column_stack(cparts), axis=1), np.max(np.column_stack(tparts), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("artifacts"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-components", type=int, default=5)
    args = ap.parse_args()
    _, raw = load_hpc(args.input)
    x = transform(raw)
    chunks = np.array_split(np.arange(len(x)), 8)
    methods = {
        "per_counter_gmm_proxy": score_independent,
        "date2024_per_counter_gmm_3sigma": score_independent,
        "joint_raw_log_gmm": score_joint,
        "conditional_gmm_pseudolikelihood": score_conditional,
    }
    records = []
    for test_block in range(len(chunks)):
        cal_block = (test_block + 1) % len(chunks)
        excluded = {test_block, cal_block}
        train_ids = np.concatenate([g for j, g in enumerate(chunks) if j not in excluded])
        cal_ids, test_ids = chunks[cal_block], chunks[test_block]
        score_cache = {}
        for name, scorer in methods.items():
            if scorer.__name__ not in score_cache:
                score_cache[scorer.__name__] = scorer(x[train_ids], x[cal_ids], x[test_ids], 100 + test_block, args.max_components)
            cal, test = score_cache[scorer.__name__]
            threshold = 3.0 if name == "date2024_per_counter_gmm_3sigma" else np.quantile(cal, 1 - args.alpha, method="higher")
            records.append({
                "test_block": test_block,
                "calibration_block": cal_block,
                "fit_n": len(train_ids),
                "calibration_n": len(cal_ids),
                "test_n": len(test_ids),
                "method": name,
                "threshold": float(threshold),
                "calibration_fpr": float(np.mean(cal > threshold)),
                "test_fpr": float(np.mean(test > threshold)),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(records)
    out.to_csv(args.output / "blocked_clean_comparison.csv", index=False)
    summary = out.groupby("method").agg(
        blocked_test_fpr_mean=("test_fpr", "mean"),
        blocked_test_fpr_std=("test_fpr", "std"),
        blocked_calibration_fpr_mean=("calibration_fpr", "mean"),
        blocks=("test_block", "count"),
    ).reset_index()
    summary.to_csv(args.output / "blocked_clean_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
