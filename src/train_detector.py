import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


COUNTERS = ["cycles", "cache-references", "LLC-loads"]


def load_hpc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # Remove an exported dataframe index only when it is an unnamed sequential column.
    for col in list(frame.columns):
        if str(col).startswith("Unnamed:"):
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy()
            if np.isfinite(values).all() and np.array_equal(values, np.arange(len(frame))):
                frame = frame.drop(columns=[col])
    missing = [c for c in COUNTERS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required HPC columns: {missing}; found {list(frame.columns)}")
    x = frame[COUNTERS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("HPC columns contain missing or non-finite values.")
    if (x <= 0).any().any():
        raise ValueError("All selected counters must be positive for log-ratio features.")
    return x


def transform(raw: pd.DataFrame) -> np.ndarray:
    cycles = raw["cycles"].to_numpy(dtype=float)
    refs = raw["cache-references"].to_numpy(dtype=float)
    llc = raw["LLC-loads"].to_numpy(dtype=float)
    return np.column_stack([
        np.log(cycles),
        np.log(refs / cycles),
        np.log(llc / refs),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and score a clean-only HPC anomaly detector.")
    parser.add_argument("--input", type=Path, required=True, help="Clean training CSV.")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--score-csv", type=Path, help="Optional CSV to score after training.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Target clean calibration tail fraction.")
    parser.add_argument("--max-components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between zero and one.")
    args.output.mkdir(parents=True, exist_ok=True)

    clean = load_hpc(args.input)
    features = transform(clean)
    fit_x, cal_x = train_test_split(features, test_size=0.25, random_state=args.seed, shuffle=True)

    scaler = StandardScaler().fit(fit_x)
    fit_z = scaler.transform(fit_x)
    cal_z = scaler.transform(cal_x)

    candidates = []
    upper = min(args.max_components, max(1, len(fit_z) // 20))
    for k in range(1, upper + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-5,
            n_init=5,
            random_state=args.seed,
        ).fit(fit_z)
        candidates.append((model.bic(fit_z), model))
    bic, detector = min(candidates, key=lambda pair: pair[0])

    cal_scores = -detector.score_samples(cal_z)
    threshold = float(np.quantile(cal_scores, 1 - args.alpha, method="higher"))
    clean_cal_fpr = float(np.mean(cal_scores > threshold))

    bundle = {
        "counters": COUNTERS,
        "feature_names": ["log_cycles", "log_cache_references_per_cycle", "log_llc_loads_per_cache_reference"],
        "scaler": scaler,
        "detector": detector,
        "threshold": threshold,
        "alpha": args.alpha,
        "selected_components": detector.n_components,
        "seed": args.seed,
        "method": "work-conditioned joint HPC density",
    }
    joblib.dump(bundle, args.output / "detector.joblib")
    summary = {
        "rows": int(len(clean)),
        "fit_rows": int(len(fit_x)),
        "calibration_rows": int(len(cal_x)),
        "selected_components": int(detector.n_components),
        "training_bic": float(bic),
        "alpha": args.alpha,
        "threshold": threshold,
        "calibration_fpr_observed": clean_cal_fpr,
        "features": bundle["feature_names"],
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.score_csv:
        score_raw = load_hpc(args.score_csv)
        scores = -detector.score_samples(scaler.transform(transform(score_raw)))
        output = score_raw.copy()
        output["anomaly_score"] = scores
        output["is_anomaly"] = scores > threshold
        output.to_csv(args.output / "scores.csv", index=False)
        print(f"Wrote {args.output / 'scores.csv'}")


if __name__ == "__main__":
    main()
