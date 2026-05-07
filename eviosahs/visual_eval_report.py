from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eviosahs.models import ensure_parent_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a visual evaluation summary report.")
    parser.add_argument("--quality-json", required=True)
    parser.add_argument("--quality-csv", required=True)
    parser.add_argument("--proxy-summary", required=True)
    parser.add_argument("--f5-metrics", required=True)
    parser.add_argument("--f5-a6-metrics", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_report(path: str | Path) -> dict[str, float]:
    return load_json(path).get("report", {})


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    quality = load_json(args.quality_json)
    quality_rows = load_csv(args.quality_csv)
    proxy_rows = load_csv(args.proxy_summary)
    f5 = metric_report(args.f5_metrics)
    f5_a6 = metric_report(args.f5_a6_metrics)

    comparison_rows = [
        [
            "F5 seven_questions",
            f5.get("Accuracy (%)", ""),
            f5.get("Sensitivity (%)", ""),
            f5.get("Specificity (%)", ""),
            f5.get("F1-Score (%)", ""),
        ],
        [
            "F5_A6 single_pass",
            f5_a6.get("Accuracy (%)", ""),
            f5_a6.get("Sensitivity (%)", ""),
            f5_a6.get("Specificity (%)", ""),
            f5_a6.get("F1-Score (%)", ""),
        ],
        [
            "Delta F5-F5_A6",
            round(float(f5.get("Accuracy (%)", 0)) - float(f5_a6.get("Accuracy (%)", 0)), 4),
            round(float(f5.get("Sensitivity (%)", 0)) - float(f5_a6.get("Sensitivity (%)", 0)), 4),
            round(float(f5.get("Specificity (%)", 0)) - float(f5_a6.get("Specificity (%)", 0)), 4),
            round(float(f5.get("F1-Score (%)", 0)) - float(f5_a6.get("F1-Score (%)", 0)), 4),
        ],
    ]

    md = "\n\n".join(
        [
            "# Visual Module Evaluation Summary",
            "## V1 Output Quality",
            md_table(
                [
                    "Anatomy Target",
                    "N",
                    "Structured Parse (%)",
                    "Fallback (%)",
                    "High Visibility (%)",
                    "Uncertain Visibility (%)",
                    "Noise (%)",
                ],
                [
                    [
                        row["anatomy_target"],
                        row["n"],
                        row["structured_parse_percent"],
                        row["fallback_percent"],
                        row["visibility_high_percent"],
                        row["visibility_uncertain_percent"],
                        row["noise_mention_percent"],
                    ]
                    for row in quality_rows
                ],
            ),
            "Overall: "
            + json.dumps(
                {
                    "total_visual_sessions": quality.get("total_visual_sessions"),
                    "structured_parse_rate_percent": quality.get("structured_parse_rate_percent"),
                    "overall_visibility_high_percent": quality.get("overall_visibility_high_percent"),
                    "overall_noise_mention_rate_percent": quality.get("overall_noise_mention_rate_percent"),
                },
                ensure_ascii=False,
            ),
            "## V3/V4 Proxy Agreement",
            md_table(
                [
                    "Proxy Task",
                    "Strong Proxy N",
                    "Covered N",
                    "Coverage (%)",
                    "Balanced Acc (%)",
                    "Macro-F1 (%)",
                    "Sensitivity (%)",
                    "Specificity (%)",
                ],
                [
                    [
                        row["proxy_task"],
                        row["strong_proxy_subset_n"],
                        row["covered_n"],
                        row["coverage_percent"],
                        row["balanced_accuracy_percent"],
                        row["macro_f1_percent"],
                        row["sensitivity_percent"],
                        row["specificity_percent"],
                    ]
                    for row in proxy_rows
                ],
            ),
            "## V5 Seven-Questions vs Single-Pass",
            md_table(["Model", "Accuracy (%)", "Sensitivity (%)", "Specificity (%)", "F1-Score (%)"], comparison_rows),
            "These outputs are proxy-consistency and quality-control analyses, not ground-truth visual accuracy.",
        ]
    )

    md_path = ensure_parent_dir(args.output_md)
    md_path.write_text(md + "\n", encoding="utf-8")

    csv_path = ensure_parent_dir(args.output_csv)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "name", "metric", "value"])
        for row in quality_rows:
            for key, value in row.items():
                if key != "anatomy_target":
                    writer.writerow(["quality", row["anatomy_target"], key, value])
        for row in proxy_rows:
            for key, value in row.items():
                if key != "proxy_task":
                    writer.writerow(["proxy", row["proxy_task"], key, value])
        for row in comparison_rows:
            for metric, value in zip(["Accuracy (%)", "Sensitivity (%)", "Specificity (%)", "F1-Score (%)"], row[1:]):
                writer.writerow(["structure_comparison", row[0], metric, value])
    print(f"Wrote visual evaluation report to {args.output_md} and {args.output_csv}")


if __name__ == "__main__":
    main()
