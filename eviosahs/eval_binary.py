from __future__ import annotations

import argparse

from eviosahs.models import compute_binary_metrics, load_prediction_records, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EviOSAHS binary predictions.")
    parser.add_argument("--pred-file", required=True, help="Path to final.jsonl or final.csv")
    parser.add_argument("--metrics-out", required=True, help="Path to write metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_prediction_records(args.pred_file)
    metrics = compute_binary_metrics(records)
    payload = {
        "pred_file": args.pred_file,
        **metrics,
    }
    write_json_atomic(payload, args.metrics_out)
    report = metrics["report"]
    print(f"Accuracy (%): {report['Accuracy (%)']:.4f}")
    print(f"Sensitivity (%): {report['Sensitivity (%)']:.4f}")
    print(f"Specificity (%): {report['Specificity (%)']:.4f}")
    print(f"F1-Score (%): {report['F1-Score (%)']:.4f}")


if __name__ == "__main__":
    main()
