import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COUNTERS = ["cycles", "cache-references", "LLC-loads"]


def read_csv(path):
    df = pd.read_csv(path)
    for col in list(df.columns):
        if str(col).startswith("Unnamed:"):
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
            if np.isfinite(vals).all() and np.array_equal(vals, np.arange(len(df))):
                df = df.drop(columns=[col])
    if not set(COUNTERS).issubset(df.columns):
        raise ValueError(f"Required columns: {COUNTERS}; found {list(df.columns)}")
    x = df[COUNTERS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError("Counter values must be finite and positive.")
    return x


def representations(x):
    logs = np.log(x)
    # Work level and two relative cache-behavior measurements.
    proposed = np.column_stack((logs[:, 0], logs[:, 1] - logs[:, 0], logs[:, 2] - logs[:, 1]))
    return logs, proposed


def fit_gmm(z, seed, max_components):
    models = []
    for k in range(1, min(max_components, max(1, len(z) // 20)) + 1):
        m = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-5,
                            n_init=3, random_state=seed).fit(z)
        models.append((m.bic(z), m))
    return min(models, key=lambda q: q[0])


def joint_scores(train, cal, test, seed, max_components):
    scaler = StandardScaler().fit(train)
    ztr, zca, zte = scaler.transform(train), scaler.transform(cal), scaler.transform(test)
    bic, model = fit_gmm(ztr, seed, max_components)
    return -model.score_samples(zca), -model.score_samples(zte), int(model.n_components), float(bic)


def independent_scores(train, cal, test, seed, max_components):
    # DATE-style per-counter density baseline; max standardized marginal NLL is the score.
    cal_parts, test_parts, selected, bics = [], [], [], []
    for j in range(train.shape[1]):
        scaler = StandardScaler().fit(train[:, [j]])
        tr, ca, te = (scaler.transform(a[:, [j]]) for a in (train, cal, test))
        bic, model = fit_gmm(tr, seed + j, max_components)
        sc, st = -model.score_samples(ca), -model.score_samples(te)
        mu, sd = float(sc.mean()), max(float(sc.std()), 1e-8)
        cal_parts.append((sc - mu) / sd)
        test_parts.append((st - mu) / sd)
        selected.append(int(model.n_components))
        bics.append(float(bic))
    return np.max(np.column_stack(cal_parts), axis=1), np.max(np.column_stack(test_parts), axis=1), selected, bics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("artifacts"))
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-components", type=int, default=5)
    args = ap.parse_args()
    x = read_csv(args.input)
    raw_log, conditional = representations(x)
    records = []
    for seed in range(args.repeats):
        idx_fit, idx_rest = train_test_split(np.arange(len(x)), test_size=0.4, random_state=seed)
        idx_cal, idx_test = train_test_split(idx_rest, test_size=0.5, random_state=1000 + seed)
        variants = {
            "independent_per_counter_gmm": (raw_log[idx_fit], raw_log[idx_cal], raw_log[idx_test], True),
            "joint_raw_log_gmm": (raw_log[idx_fit], raw_log[idx_cal], raw_log[idx_test], False),
            "work_conditioned_feature_gmm": (conditional[idx_fit], conditional[idx_cal], conditional[idx_test], False),
        }
        for name, (tr, ca, te, independent) in variants.items():
            if independent:
                sc, st, comps, bics = independent_scores(tr, ca, te, seed, args.max_components)
                component_note, bic_note = json.dumps(comps), json.dumps(bics)
            else:
                sc, st, comp, bic = joint_scores(tr, ca, te, seed, args.max_components)
                component_note, bic_note = str(comp), str(bic)
            threshold = float(np.quantile(sc, 1 - args.alpha, method="higher"))
            records.append({
                "seed": seed,
                "method": name,
                "fit_n": int(len(idx_fit)),
                "calibration_n": int(len(idx_cal)),
                "test_n": int(len(idx_test)),
                "threshold": threshold,
                "calibration_fpr": float(np.mean(sc > threshold)),
                "test_fpr": float(np.mean(st > threshold)),
                "selected_components": component_note,
                "training_bic": bic_note,
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
