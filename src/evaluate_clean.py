import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from detector_core import conditional_nll, fit_gmm, load_hpc, transform


def fit_score(train, cal, test, seed, max_components):
    scaler, model, bic = fit_gmm(train, seed, max_components)
    cz, vz = scaler.transform(cal), scaler.transform(test)
    cal_scores = -model.score_samples(cz)
    test_scores = -model.score_samples(vz)
    return cal_scores, test_scores, str(model.n_components), str(bic)


def fit_conditional(train, cal, test, seed, max_components):
    scaler, model, bic = fit_gmm(train, seed, max_components)
    cz, vz = scaler.transform(cal), scaler.transform(test)
    cal_scores = conditional_nll(cz, model).mean(axis=1)
    test_scores = conditional_nll(vz, model).mean(axis=1)
    return cal_scores, test_scores, str(model.n_components), str(bic)


def fit_independent(train, cal, test, seed, max_components):
    cal_parts, test_parts, comps, bics = [], [], [], []
    for j in range(train.shape[1]):
        ca, te, comp, bic = fit_score(
            train[:, [j]], cal[:, [j]], test[:, [j]], seed + j, max_components
        )
        mu, sd = float(ca.mean()), max(float(ca.std()), 1e-8)
        cal_parts.append((ca - mu) / sd)
        test_parts.append((te - mu) / sd)
        comps.append(comp)
        bics.append(bic)
    return np.max(np.column_stack(cal_parts), axis=1), np.max(np.column_stack(test_parts), axis=1), str(comps), str(bics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("artifacts"))
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-components", type=int, default=5)
    args = ap.parse_args()
    _, raw = load_hpc(args.input)
    x = transform(raw)
    records = []
    for seed in range(args.repeats):
        idx_fit, idx_rest = train_test_split(np.arange(len(x)), test_size=0.4, random_state=seed)
        idx_cal, idx_test = train_test_split(idx_rest, test_size=0.5, random_state=1000 + seed)
        fit, cal, test = x[idx_fit], x[idx_cal], x[idx_test]
        methods = [
            ("independent_per_counter_gmm_proxy", fit_independent(fit, cal, test, seed, args.max_components)),
            ("joint_raw_log_gmm", fit_score(fit, cal, test, seed, args.max_components)),
            ("conditional_gmm_pseudolikelihood", fit_conditional(fit, cal, test, seed, args.max_components)),
        ]
        for name, (cal_scores, test_scores, components, bic) in methods:
            threshold = float(np.quantile(cal_scores, 1 - args.alpha, method="higher"))
            records.append({
                "seed": seed,
                "method": name,
                "fit_n": len(idx_fit),
                "calibration_n": len(idx_cal),
                "test_n": len(idx_test),
                "threshold": threshold,
                "calibration_fpr": float(np.mean(cal_scores > threshold)),
                "test_fpr": float(np.mean(test_scores > threshold)),
                "selected_components": components,
                "training_bic": bic,
            })
    args.output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results.to_csv(args.output / "clean_only_comparison.csv", index=False)
    summary = results.groupby("method").agg(
        test_fpr_mean=("test_fpr", "mean"),
        test_fpr_std=("test_fpr", "std"),
        calibration_fpr_mean=("calibration_fpr", "mean"),
        repeats=("seed", "count"),
    ).reset_index()
    summary.to_csv(args.output / "clean_only_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nAttack TPR/F1/AUROC are not computable from clean-only traces.")


if __name__ == "__main__":
    main()
