import argparse
import textwrap
from pathlib import Path

import pandas as pd


def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages, dest):
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    refs = " ".join(f"{3 + 2*i} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{refs}] /Count {len(pages)} >>".encode("ascii"))
    font_id = 3 + 2*len(pages)
    for i, lines in enumerate(pages):
        page_id, stream_id = 3 + 2*i, 4 + 2*i
        commands = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line in lines:
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
    lines = [
        "IIT Kharagpur ML Security Hackathon - Problem 1",
        "Clean-Only HPC Backdoor Anomaly Detection",
        "",
        "Objective",
        "Detect inference traces that differ from benign HPC behavior, with no",
        "backdoor traces available during training. The supplied file has 800 clean",
        "inferences and three hardware performance counters.",
        "",
        "Data preparation",
        "Features: cycles, cache-references, LLC-loads. The unnamed sequential",
        "column is a row index and is dropped. Features are log(cycles),",
        "log(cache-references/cycles), and log(LLC-loads/cache-references).",
        "Cache references and LLC loads have approximately 0.92 Pearson correlation",
        "in the supplied clean traces.",
        "",
        "Proposed detector",
        "A Gaussian mixture density is fitted to the joint transformed HPC vector.",
        "BIC selects the mixture size on the fitting split. The anomaly score is",
        "negative log likelihood. A separate clean calibration split sets the",
        "threshold at the configured 95th percentile. The artifact saves scaler,",
        "model, feature names, and threshold.",
        "",
        "Evaluation design",
        "Twenty-five deterministic clean-only splits: 60% fit, 20% calibration,",
        "20% held-out clean evaluation. Baselines: (1) per-counter GMM proxy using",
        "the maximum standardized marginal negative log likelihood, and (2) joint",
        "GMM on log raw counter values. The proposed work-conditioned feature GMM",
        "is compared using identical splits.",
        "",
        "Clean-only results: mean held-out FPR +/- standard deviation",
        f"Per-counter GMM proxy: {val('independent_per_counter_gmm','test_fpr_mean'):.2%} +/- {val('independent_per_counter_gmm','test_fpr_std'):.2%}",
        f"Joint raw-log GMM: {val('joint_raw_log_gmm','test_fpr_mean'):.2%} +/- {val('joint_raw_log_gmm','test_fpr_std'):.2%}",
        f"Work-conditioned feature GMM: {val('work_conditioned_feature_gmm','test_fpr_mean'):.2%} +/- {val('work_conditioned_feature_gmm','test_fpr_std'):.2%}",
        "",
        "Interpretation and limits",
        "These results measure clean false alarms only. Small FPR differences do",
        "not demonstrate superior backdoor detection. No attack examples were",
        "provided, so TPR, attack FPR, F1, and AUROC cannot be calculated. Triggered",
        "proxy runs, if collected, must be reported as separate evaluation.",
        "Calibration assumes future clean traces are comparable to calibration",
        "traces; it does not guarantee the same FPR under distribution shift.",
        "",
        "Related work",
        "Alam, Wang, Maniatakos, DATE 2024, Detecting Backdoor Attacks in Black-Box",
        "Neural Networks through Hardware Performance Counters.",
        "https://past.date-conference.com/proceedings-archive/2024/DATA/1142_pdf_upload.pdf",
        "",
        "Reproduction commands",
        "pip install -r requirements.txt",
        "python src/train_detector.py --input data/clean_hpc.csv --output artifacts",
        "python src/evaluate_clean.py --input data/clean_hpc.csv --output artifacts",
    ]
    expanded = []
    for line in lines:
        expanded.extend(textwrap.wrap(line, width=88) or [""])
    pages = [expanded[i:i+48] for i in range(0, len(expanded), 48)]
    make_pdf(pages, args.output)
    print(f"Wrote {args.output} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
