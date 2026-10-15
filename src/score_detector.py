import argparse
from pathlib import Path

import joblib
import numpy as np

from detector_core import COUNTERS, conditional_nll, load_hpc, transform


def main():
    ap = argparse.ArgumentParser(description="Score HPC traces with the saved detector.")
    ap.add_argument("--model", type=Path, default=Path("artifacts/detector.joblib"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("artifacts/scores.csv"))
    args = ap.parse_args()
    bundle = joblib.load(args.model)
    source, raw = load_hpc(args.input)
    z = bundle["scaler"].transform(transform(raw))
    per_counter = conditional_nll(z, bundle["detector"])
    scores = per_counter.mean(axis=1)
    result = source.copy()
    result["anomaly_score"] = scores
    result["most_unusual_conditional_counter"] = np.asarray(COUNTERS)[np.argmax(per_counter, axis=1)]
    result["is_anomaly"] = scores > float(bundle["threshold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} rows; threshold={bundle['threshold']:.6f}; output={args.output}")


if __name__ == "__main__":
    main()
