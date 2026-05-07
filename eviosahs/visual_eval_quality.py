from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eviosahs.models import VISUAL_NOISE_KEYWORDS, read_jsonl, write_csv, write_json_atomic


CSV_FIELDS = [
    "anatomy_target",
    "n",
    "structured_parse_percent",
    "fallback_percent",
    "visibility_high_percent",
    "visibility_medium_percent",
    "visibility_uncertain_percent",
    "noise_mention_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute question-level visual output quality metrics.")
    parser.add_argument("--rows-jsonl", required=True, help="Question-level rows from visual_eval_prepare.py")
    parser.add_argument("--output-csv", required=True, help="Per-question quality CSV")
    parser.add_argument("--output-json", required=True, help="Overall quality summary JSON")
    return parser.parse_args()


def pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 4) if total else 0.0


def has_noise(row: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(row.get("raw_visual_response") or ""),
            str(row.get("visual_observation") or ""),
        ]
    ).lower()
    return any(keyword in text for keyword in VISUAL_NOISE_KEYWORDS)


def compute_quality(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target[str(row.get("anatomy_target") or "unknown")].append(row)

    table: list[dict[str, Any]] = []
    total_counter = Counter()
    for target in sorted(by_target):
        target_rows = by_target[target]
        n = len(target_rows)
        structured = sum(str(row.get("visual_parse_status")).lower() == "structured" for row in target_rows)
        fallback = n - structured
        high = sum(str(row.get("visibility")).lower() == "high" for row in target_rows)
        medium = sum(str(row.get("visibility")).lower() == "medium" for row in target_rows)
        uncertain = sum(str(row.get("visibility")).lower() == "uncertain" for row in target_rows)
        noise = sum(has_noise(row) for row in target_rows)
        table.append(
            {
                "anatomy_target": target,
                "n": n,
                "structured_parse_percent": pct(structured, n),
                "fallback_percent": pct(fallback, n),
                "visibility_high_percent": pct(high, n),
                "visibility_medium_percent": pct(medium, n),
                "visibility_uncertain_percent": pct(uncertain, n),
                "noise_mention_percent": pct(noise, n),
            }
        )
        total_counter.update(
            {
                "n": n,
                "structured": structured,
                "fallback": fallback,
                "high": high,
                "medium": medium,
                "uncertain": uncertain,
                "noise": noise,
            }
        )

    n_total = int(total_counter["n"])
    summary = {
        "total_visual_sessions": n_total,
        "structured_parse_rate_percent": pct(int(total_counter["structured"]), n_total),
        "fallback_rate_percent": pct(int(total_counter["fallback"]), n_total),
        "overall_visibility_high_percent": pct(int(total_counter["high"]), n_total),
        "overall_visibility_medium_percent": pct(int(total_counter["medium"]), n_total),
        "overall_visibility_uncertain_percent": pct(int(total_counter["uncertain"]), n_total),
        "overall_noise_mention_rate_percent": pct(int(total_counter["noise"]), n_total),
        "by_question": table,
    }
    return table, summary


def main() -> None:
    args = parse_args()
    table, summary = compute_quality(read_jsonl(Path(args.rows_jsonl)))
    write_csv(table, args.output_csv, CSV_FIELDS)
    write_json_atomic(summary, args.output_json)
    print(f"Wrote quality table to {args.output_csv}")
    print(f"Wrote quality summary to {args.output_json}")


if __name__ == "__main__":
    main()
