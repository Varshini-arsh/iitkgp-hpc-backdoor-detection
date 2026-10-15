# IITK HPC Backdoor Detection

Problem 1: clean-only anomaly detection for backdoor-triggered neural-network inference using hardware performance counters.

## Method

The proposed detector fits a small Gaussian mixture density to three transformed HPC features:

- log(cycles) for inference work
- log(cache-references / cycles) for relative cache activity
- log(LLC-loads / cache-references) for LLC activity relative to cache references

The mixture size is selected by BIC using only the fitting split. The anomaly score is negative log likelihood. A held-out clean calibration split sets a 5% tail threshold. The trained bundle stores preprocessing, model, feature names, and threshold.

This is an empirical alternative to the closest published HPC backdoor detector, which fits per-counter GMMs. It is not a claim that the underlying ingredients are globally novel or proven to improve attack detection.

## Data

data/clean_hpc.csv contains 800 clean traces and three HPC columns: cache-references, cycles, and LLC-loads. Its unnamed column is a sequential row index and is dropped automatically. In these clean traces, cache references and LLC loads have Pearson correlation approximately 0.92.

## Setup and training

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/train_detector.py --input data/clean_hpc.csv --output artifacts

This writes artifacts/detector.joblib and artifacts/training_summary.json.

## Clean-only comparison

    python src/evaluate_clean.py --input data/clean_hpc.csv --output artifacts --repeats 25

It compares a per-counter GMM proxy baseline, a joint GMM on log raw counters, and the work-conditioned feature GMM. Results are in artifacts/clean_only_comparison.csv and artifacts/clean_only_summary.csv.

On the current 25-split run, mean held-out clean FPR was 5.33% for the per-counter proxy, 4.60% for the raw joint GMM, and 4.75% for the proposed features. These clean-only results do not measure backdoor detection and do not establish superiority.

## Scoring new traces

    python src/score_detector.py --model artifacts/detector.joblib --input data/to_score.csv --output artifacts/scores.csv

The scoring CSV must include the three named counters. Output scores and labels are written to artifacts/scores.csv.

## Limitations

The supplied data contains no backdoor traces. TPR, attack FPR, F1, and AUROC are unavailable until there is a separately labeled attack evaluation set. Locally generated triggered traces are proxy experiments and must not be used for fitting or calibration. The clean calibration threshold assumes future clean traces are comparable to calibration data; it is not a guarantee under distribution shift.

## Related work

Alam, Wang, and Maniatakos, “Detecting Backdoor Attacks in Black-Box Neural Networks through Hardware Performance Counters,” DATE 2024. Their public method is included as a baseline reference; this repository does not copy their implementation.

## Report

See reports/method_report.pdf for method, split design, clean-only results, and limitations.

