import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

COUNTERS = ["cycles", "cache-references", "LLC-loads"]


def load_input(path):
    df = pd.read_csv(path)
    for col in list(df.columns):
        if str(col).startswith("Unnamed:"):
            values = pd.to_numeric(df[col], errors="coerce").to_numpy()
            if np.isfinite(values).all() and np.array_equal(values, np.arange(len(df))):
                df = df.drop(columns=[col])
    missing = [c for c in COUNTERS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required HPC columns: {missing}")
    raw = df[COUNTERS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(raw.to_numpy()).all() or (raw <= 0).any().any():
        raise ValueError("Counter values must be finite and positive.")
    return df, raw


def transform(raw):
    cycles = raw["cycles"].to_numpy(float)
    refs = raw["cache-references"].to_numpy(float)
    llc = raw["LLC-loads"].to_numpy(float)
    return np.column_stack((np.log(cycles), np.log(refs / cycles), np.log(llc / refs)))


def main():
    ap = argparse.ArgumentParser(description="Score HPC traces with the saved detector.")
    ap.add_argument("--model", type=Path, default=Path("artifacts/detector.joblib"))
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("artifacts/scores.csv"))
    args = ap.parse_args()
    bundle = joblib.load(args.model)
    source, raw = load_input(args.input)
    features = bundle["scaler"].transform(transform(raw))
    scores = -bundle["detector"].score_samples(features)
    result = source.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = scores > float(bundle["threshold"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} rows; threshold={bundle['threshold']:.6f}; output={args.output}")


if __name__ == "__main__":
    main()
