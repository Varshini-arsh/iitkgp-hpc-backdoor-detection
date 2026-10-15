import argparse
import textwrap
from pathlib import Path
import pandas as pd


def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(lines, dest):
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=88) or [""])
    pages = [wrapped[i:i+48] for i in range(0, len(wrapped), 48)]
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2*i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))
    font_id = 3 + 2 * len(pages)
    for i, page_lines in enumerate(pages):
        page_id, stream_id = 3 + 2*i, 4 + 2*i
        commands = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line in page_lines:
            commands.extend([f"({esc(line)}) Tj", "T*"])
        commands.append("ET")
        stream = "\n".join(commands).encode("ascii", errors="replace")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {stream_id} 0 R >>".encode("ascii"))
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pdf, offsets = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("ascii"))
        pdf.extend(obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("reports/method_report.pdf"))
    args = ap.parse_args()
    df = pd.read_csv(args.summary).set_index("method")
    def val(method, col):
        return float(df.loc[method, col])
    blocked = pd.read_csv(args.summary.with_name("blocked_clean_summary.csv")).set_index("method")
    def bval(method, col):
        return float(blocked.loc[method, col])
    lines = [
        "IIT Kharagpur ML Security Hackathon - Problem 1",
        "Conditional HPC Anomaly Detection for Backdoor Inference",
        "",
        "Objective",
        "Learn normal inference behavior from clean HPC traces and flag deviations.",
        "The supplied data has 800 clean inferences and no attack examples.",
        "",
        "Data",
        "Counters: cycles, cache-references, LLC-loads. The unnamed sequential",
        "column is a row index and is dropped. Cache references and LLC loads have",
        "approximately 0.92 Pearson correlation in the supplied clean traces.",
        "",
        "Proposed method",
        "A full-covariance Gaussian mixture models the joint distribution of the",
        "three log-transformed counters. For each counter, calculate its conditional",
        "negative log likelihood given the other two counters. Average the three",
        "conditional scores for the final anomaly score. This pseudo-likelihood",
        "measures whether each counter is consistent with the clean relationships",
        "learned from the other counters. Select mixture size by BIC on fitting data.",
        "Calibrate the final threshold on separate clean traces.",
        "",
        "Evaluation",
        "25 deterministic clean-only splits; each uses 60% fit, 20% calibration,",
        "and 20% held-out clean evaluation. Baselines are a per-counter GMM proxy",
        "and a joint GMM on log raw counts.",
        "",
        "Mean held-out clean FPR +/- standard deviation",
        f"DATE-style per-counter GMM (3-sigma): {val('date2024_per_counter_gmm_3sigma','test_fpr_mean'):.2%} +/- {val('date2024_per_counter_gmm_3sigma','test_fpr_std'):.2%}",
        f"Per-counter GMM proxy at 5% calibration: {val('independent_per_counter_gmm_proxy','test_fpr_mean'):.2%} +/- {val('independent_per_counter_gmm_proxy','test_fpr_std'):.2%}",
        f"Joint raw-log GMM: {val('joint_raw_log_gmm','test_fpr_mean'):.2%} +/- {val('joint_raw_log_gmm','test_fpr_std'):.2%}",
        f"Conditional GMM pseudo-likelihood: {val('conditional_gmm_pseudolikelihood','test_fpr_mean'):.2%} +/- {val('conditional_gmm_pseudolikelihood','test_fpr_std'):.2%}",
        "",
        "Contiguous-block stress check",
        "Eight contiguous blocks of 100 traces were held out in turn. The next",
        "block calibrated the threshold; the other six blocks fitted each model.",
        f"Conditional pseudo-likelihood clean FPR: {bval('conditional_gmm_pseudolikelihood','blocked_test_fpr_mean'):.2%} +/- {bval('conditional_gmm_pseudolikelihood','blocked_test_fpr_std'):.2%}.",
        f"DATE-style per-counter GMM (3-sigma): {bval('date2024_per_counter_gmm_3sigma','blocked_test_fpr_mean'):.2%} +/- {bval('date2024_per_counter_gmm_3sigma','blocked_test_fpr_std'):.2%}.",
        f"Per-counter GMM proxy at 5% calibration: {bval('per_counter_gmm_proxy','blocked_test_fpr_mean'):.2%} +/- {bval('per_counter_gmm_proxy','blocked_test_fpr_std'):.2%}.",
        f"Joint raw-log GMM: {bval('joint_raw_log_gmm','blocked_test_fpr_mean'):.2%} +/- {bval('joint_raw_log_gmm','blocked_test_fpr_std'):.2%}.",
        "No method has a demonstrated advantage for attack detection, and this",
        "single-file experiment cannot establish broad robustness.",
        "",
        "Interpretation and limits",
        "These results assess clean false alarms only; they do not establish attack",
        "detection performance. No backdoor traces were provided, so TPR, attack",
        "FPR, F1, and AUROC cannot be calculated. Proxy-trigger runs, if collected,",
        "must be reported separately and kept out of fitting and calibration.",
        "Calibration assumes future clean traces are exchangeable with calibration",
        "traces; it does not guarantee the same false-positive rate under shift.",
        "The conditional scoring design differs from the per-counter GMM design in",
        "the closest published HPC backdoor-detection paper, but this is not a claim",
        "that conditional likelihood or GMMs are new in general.",
        "",
        "Related work",
        "Alam, Wang, Maniatakos, DATE 2024, Detecting Backdoor Attacks in Black-Box",
        "Neural Networks through Hardware Performance Counters.",
        "https://past.date-conference.com/proceedings-archive/2024/DATA/1142_pdf_upload.pdf",
        "",
        "Reproduce: pip install -r requirements.txt",
        "python src/train_detector.py --input data/clean_hpc.csv --output artifacts",
        "python src/evaluate_clean.py --input data/clean_hpc.csv --output artifacts",
        "python src/evaluate_blocked.py --input data/clean_hpc.csv --output artifacts",
    ]
    make_pdf(lines, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
