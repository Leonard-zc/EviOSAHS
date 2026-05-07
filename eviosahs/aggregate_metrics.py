from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from eviosahs.models import compute_binary_metrics, load_prediction_records, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate fold metrics for EviOSAHS experiments.")
    parser.add_argument("--input-root", required=True, help="Directory containing fold_* subdirectories")
    parser.add_argument("--mode", choices=["metrics_json", "pred_file"], default="metrics_json", help="Aggregate from metrics.json or directly from prediction files")
    parser.add_argument("--pred-filename", default="final.jsonl", help="Prediction filename under each fold directory when mode=pred_file")
    parser.add_argument("--metrics-filename", default="metrics.json", help="Metrics filename under each fold directory when mode=metrics_json")
    parser.add_argument("--output-json", required=True, help="Output summary json path")
    return parser.parse_args()


METRIC_KEYS = ["Accuracy (%)", "Sensitivity (%)", "Specificity (%)", "F1-Score (%)"]


def collect_fold_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("fold_")])


def load_fold_metrics(fold_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "metrics_json":
        payload = json.loads((fold_dir / args.metrics_filename).read_text(encoding="utf-8"))
        return payload["report"] if "report" in payload else payload["metrics"]["report"]

    records = load_prediction_records(fold_dir / args.pred_filename)
    return compute_binary_metrics(records)["report"]


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "mean": round(mean(values), 4) if values else 0.0,
        "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    fold_dirs = collect_fold_dirs(root)
    summaries: dict[str, Any] = {}
    fold_reports: list[dict[str, Any]] = []

    for fold_dir in fold_dirs:
        report = load_fold_metrics(fold_dir, args)
        fold_reports.append({"fold": fold_dir.name, "report": report})

    for key in METRIC_KEYS:
        values = [float(item["report"][key]) for item in fold_reports]
        summaries[key] = summarize(values)

    payload = {
        "input_root": str(root.resolve()),
        "mode": args.mode,
        "fold_count": len(fold_reports),
        "fold_reports": fold_reports,
        "summary": summaries,
    }
    write_json_atomic(payload, args.output_json)
    for key in METRIC_KEYS:
        print(f"{key}: mean={summaries[key]['mean']:.4f}, std={summaries[key]['std']:.4f}")


if __name__ == "__main__":
    main()
