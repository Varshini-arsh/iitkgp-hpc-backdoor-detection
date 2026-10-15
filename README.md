# IITK HPC Backdoor Detection

Problem 1: clean-only anomaly detection for backdoor-triggered neural-network inference using hardware performance counters.

## Proposed method

A full-covariance Gaussian mixture models the joint distribution of the three log-transformed HPCs. For each counter, the detector computes its conditional negative log likelihood given the other two counters. The final anomaly score is the mean of the three conditional scores. This pseudo-likelihood score measures whether each counter is consistent with the relationships learned from the clean traces.

The mixture size is selected by BIC using only the fitting split. A separate clean calibration split sets the alert threshold. The saved detector contains the model, scaler, feature names, and threshold. The scoring script also reports the counter with the largest conditional score.

This is a distinct scoring rule from the closest published HPC backdoor method, which fits separate per-counter GMMs, and from ordinary joint-GMM negative log likelihood. It is not a claim that conditional likelihood or Gaussian mixtures are new in general. Attack detection performance remains unverified without backdoor traces.

## Data

data/clean_hpc.csv contains 800 clean traces with counters cache-references, cycles, and LLC-loads. The unnamed sequential column is an exported row index and is removed. In the clean traces, cache-references and LLC-loads have Pearson correlation approximately 0.92.

## Setup and train

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python src/train_detector.py --input data/clean_hpc.csv --output artifacts

This writes artifacts/detector.joblib and artifacts/training_summary.json.

## Evaluate clean false alarms

    python src/evaluate_clean.py --input data/clean_hpc.csv --output artifacts --repeats 25

The script compares a per-counter GMM proxy, joint GMM on log raw counters, and the conditional pseudo-likelihood detector. Results are saved to artifacts/clean_only_comparison.csv and artifacts/clean_only_summary.csv.

Current 25-split mean held-out clean FPR:
- Per-counter GMM proxy: 5.35% (standard deviation 2.21 percentage points)
- Joint raw-log GMM: 4.60% (standard deviation 1.69 percentage points)
- Conditional GMM pseudo-likelihood: 4.30% (standard deviation 2.08 percentage points)

These results assess clean false alarms only. They do not show which model best detects backdoor traces.

## Score traces with the saved model

    python src/score_detector.py --model artifacts/detector.joblib --input data/to_score.csv --output artifacts/scores.csv

The input CSV must include the three counter columns. Output includes the anomaly score, label, and most unusual conditional counter.

## Limits

No backdoor traces were supplied. TPR, attack FPR, F1, and AUROC cannot be measured yet. Any locally generated triggered traces should be used only as separate proxy evaluation and kept out of model fitting and threshold calibration. Clean calibration assumes new clean traces are exchangeable with calibration data and does not guarantee the same FPR under distribution shift.

## Related work

Alam, Wang, and Maniatakos, “Detecting Backdoor Attacks in Black-Box Neural Networks through Hardware Performance Counters,” DATE 2024. https://past.date-conference.com/proceedings-archive/2024/DATA/1142_pdf_upload.pdf

## Report

See reports/method_report.pdf for method, split design, clean-only comparison, and limitations.
