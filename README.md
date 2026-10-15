# IITK HPC Backdoor Detection

Problem 1: clean-only anomaly detection for backdoor-triggered neural-network inference using hardware performance counters.

## Hackathon problem statement

**Problem 1: Detection of Backdoor Attacks using HPCs**

An untrusted party may train a model to behave normally on most inputs but change its prediction when a special trigger pattern is embedded in an input. Teams receive hardware performance counter (HPC) traces from benign inferences and must train an anomaly detector that distinguishes traces from backdoor inferences from clean traces.

The provided CSV contains HPC columns and 800 rows, one per clean inference. No backdoor inference traces are provided, preserving the strict anomaly-detection setting.

**Requested repository deliverables:** complete code with finalized parameters, a saved detector model, a README, and a short PDF report describing the method.

**Evaluation:** the organizer's private validation set contains clean and backdoor traces. The stated metrics include accuracy, true-positive rate (TPR), false-positive rate (FPR), F1, and AUROC. Those attack-set metrics cannot be measured from the supplied clean-only training CSV; this repository reports clean false-alarm evaluation and makes that limitation explicit.

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

The script compares the DATE 2024-style per-counter GMM with a 3-sigma threshold, a per-counter GMM calibrated to 5% FPR, a joint GMM on log raw counters, and the conditional pseudo-likelihood detector. The DATE-style baseline estimates mean and standard deviation on the separate calibration split so all test FPRs are measured on held-out clean rows. Results are saved to artifacts/clean_only_comparison.csv and artifacts/clean_only_summary.csv.

Current 25-split mean held-out clean FPR:
- DATE-style per-counter GMM, 3-sigma rule: 3.85% (standard deviation 1.85 percentage points)
- Per-counter GMM proxy, calibrated to 5%: 5.35% (standard deviation 2.21 points)
- Joint raw-log GMM, calibrated to 5%: 4.60% (standard deviation 1.69 points)
- Conditional GMM pseudo-likelihood, calibrated to 5%: 4.30% (standard deviation 2.08 points)

The published-style baseline has the lowest clean FPR in these splits. The conditional score is retained as the proposed dependency-aware method, but this clean-only experiment does not show that it detects backdoors better.

## Contiguous-block stress check

`python src/evaluate_blocked.py --input data/clean_hpc.csv --output artifacts` holds out each of eight contiguous 100-trace blocks in turn and calibrates on the following block. The remaining six blocks are used for fitting. Mean held-out clean FPR was 4.38% for the DATE-style per-counter GMM 3-sigma rule (SD 2.62 percentage points), 4.50% for conditional pseudo-likelihood (SD 3.02 points), 4.63% for the calibrated per-counter GMM proxy (SD 2.67 points), and 4.88% for joint raw-log GMM (SD 4.12 points). The conditional method does not lead on this clean-only metric. This single-file check cannot establish attack detection or broad robustness. Results are in `artifacts/blocked_clean_summary.csv`.

## Score traces with the saved model

    python src/score_detector.py --model artifacts/detector.joblib --input data/to_score.csv --output artifacts/scores.csv

The input CSV must include the three counter columns. Output includes the anomaly score, label, and most unusual conditional counter.

## Limits

No backdoor traces were supplied. TPR, attack FPR, F1, and AUROC cannot be measured yet. Any locally generated triggered traces should be used only as separate proxy evaluation and kept out of model fitting and threshold calibration. Clean calibration assumes new clean traces are exchangeable with calibration data and does not guarantee the same FPR under distribution shift.

## Related work

Alam, Wang, and Maniatakos, “Detecting Backdoor Attacks in Black-Box Neural Networks through Hardware Performance Counters,” DATE 2024. https://past.date-conference.com/proceedings-archive/2024/DATA/1142_pdf_upload.pdf

## Report

See reports/method_report.pdf for method, split design, clean-only comparison, and limitations.
