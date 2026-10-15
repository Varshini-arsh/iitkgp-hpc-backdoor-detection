import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from detector_core import COUNTERS, FEATURES, calibrated_threshold, conditional_nll, fit_gmm, load_hpc, transform


def main():
    ap = argparse.ArgumentParser(description="Train a clean-only conditional HPC anomaly detector.")
    ap.add_argument("--input", type=Path, required=True, help="Clean training CSV.")
    ap.add_argument("--output", type=Path, default=Path("artifacts"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-components", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between zero and one.")
    args.output.mkdir(parents=True, exist_ok=True)
    _, raw = load_hpc(args.input)
    features = transform(raw)
    fit_x, cal_x = train_test_split(features, test_size=0.25, random_state=args.seed, shuffle=True)
    scaler, detector, bic = fit_gmm(fit_x, args.seed, args.max_components)
    calibration_parts = conditional_nll(scaler.transform(cal_x), detector)
    calibration_scores = calibration_parts.mean(axis=1)
    threshold = calibrated_threshold(calibration_scores, args.alpha)
    observed_fpr = float(np.mean(calibration_scores > threshold))

    bundle = {
        "counters": COUNTERS,
        "feature_names": FEATURES,
        "scaler": scaler,
        "detector": detector,
        "threshold": threshold,
        "alpha": args.alpha,
        "selected_components": detector.n_components,
        "seed": args.seed,
        "score_type": "mean conditional negative log likelihood",
        "method": "conditional HPC density",
    }
    joblib.dump(bundle, args.output / "detector.joblib")
    summary = {
        "rows": int(len(features)),
        "fit_rows": int(len(fit_x)),
        "calibration_rows": int(len(cal_x)),
        "selected_components": int(detector.n_components),
        "training_bic": bic,
        "alpha": args.alpha,
        "threshold": threshold,
        "calibration_fpr_observed": observed_fpr,
        "feature_names": FEATURES,
        "score_type": bundle["score_type"],
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
