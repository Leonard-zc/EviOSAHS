from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from eviosahs.models import read_jsonl, write_csv, write_json_atomic


SUMMARY_FIELDS = [
    "proxy_task",
    "strong_proxy_subset_n",
    "covered_n",
    "coverage_percent",
    "balanced_accuracy_percent",
    "macro_f1_percent",
    "sensitivity_percent",
    "specificity_percent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute weak proxy agreement for visual outputs.")
    parser.add_argument("--rows-jsonl", required=True, help="Question-level rows from visual_eval_prepare.py")
    parser.add_argument("--output-dir", required=True, help="Directory for proxy JSON/CSV outputs")
    return parser.parse_args()


def pct(value: float) -> float:
    return round(100.0 * value, 4)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def binary_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    tp = sum(row["proxy_label"] == "positive" and row["visual_label"] == "positive" for row in rows)
    tn = sum(row["proxy_label"] == "negative" and row["visual_label"] == "negative" for row in rows)
    fp = sum(row["proxy_label"] == "negative" and row["visual_label"] == "positive" for row in rows)
    fn = sum(row["proxy_label"] == "positive" and row["visual_label"] == "negative" for row in rows)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    pos_precision = tp / (tp + fp) if tp + fp else 0.0
    neg_precision = tn / (tn + fn) if tn + fn else 0.0
    pos_f1 = f1(pos_precision, sensitivity)
    neg_f1 = f1(neg_precision, specificity)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity_percent": pct(sensitivity),
        "specificity_percent": pct(specificity),
        "balanced_accuracy_percent": pct(mean([sensitivity, specificity])),
        "macro_f1_percent": pct(mean([pos_f1, neg_f1])),
    }


def neck_visual_label(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["not clearly visible", "unclear", "not visible"]):
        return "uncovered"
    if any(token in lowered for token in ["average", "normal", "within normal"]):
        return "negative"
    if re.search(r"\b(thick|large|prominent)\b", lowered):
        return "positive"
    return "uncovered"


def fat_visual_label(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["no visible", "normal amount", "no significant"]):
        return "negative"
    if any(token in lowered for token in ["minimal", "mild excess", "some visible"]):
        return "uncovered"
    if any(token in lowered for token in ["excess", "noticeable", "moderate amount", "double chin", "puffiness", "fat accumulation"]):
        return "positive"
    return "uncovered"


def build_neck_proxy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    neck_rows = [row for row in rows if row.get("anatomy_target") == "neck" and row.get("neck_circumference_cm") not in (None, "")]
    by_sex: dict[str, list[float]] = defaultdict(list)
    for row in neck_rows:
        by_sex[str(row.get("sex") or "unknown")].append(float(row["neck_circumference_cm"]))
    cutoffs = {
        sex: {"q1": percentile(values, 0.25), "q3": percentile(values, 0.75)}
        for sex, values in by_sex.items()
    }
    strong_rows: list[dict[str, str]] = []
    uncovered = 0
    for row in neck_rows:
        sex = str(row.get("sex") or "unknown")
        value = float(row["neck_circumference_cm"])
        if value <= cutoffs[sex]["q1"]:
            proxy_label = "negative"
        elif value >= cutoffs[sex]["q3"]:
            proxy_label = "positive"
        else:
            continue
        visual_label = neck_visual_label(str(row.get("visual_observation") or ""))
        if visual_label == "uncovered":
            uncovered += 1
        else:
            strong_rows.append({"proxy_label": proxy_label, "visual_label": visual_label})
    metrics = binary_metrics(strong_rows)
    subset_n = len(strong_rows) + uncovered
    return {
        "proxy_task": "neck_vs_neck_circumference",
        "cutoffs": cutoffs,
        "strong_proxy_subset_n": subset_n,
        "covered_n": len(strong_rows),
        "coverage_percent": pct(len(strong_rows) / subset_n) if subset_n else 0.0,
        **metrics,
    }


def build_fat_proxy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fat_rows = [row for row in rows if row.get("anatomy_target") == "face_and_neck_fat"]
    strong_rows: list[dict[str, str]] = []
    uncovered = 0
    for row in fat_rows:
        bmi_category = str(row.get("bmi_category") or "").lower()
        whr_category = str(row.get("waist_hip_ratio_category") or "").lower()
        if "obesity" in bmi_category or whr_category in {"elevated", "markedly elevated"}:
            proxy_label = "positive"
        elif bmi_category == "healthy weight" and whr_category == "not elevated":
            proxy_label = "negative"
        else:
            continue
        visual_label = fat_visual_label(str(row.get("visual_observation") or ""))
        if visual_label == "uncovered":
            uncovered += 1
        else:
            strong_rows.append({"proxy_label": proxy_label, "visual_label": visual_label})
    metrics = binary_metrics(strong_rows)
    subset_n = len(strong_rows) + uncovered
    return {
        "proxy_task": "face_and_neck_fat_vs_bmi_whr",
        "strong_proxy_subset_n": subset_n,
        "covered_n": len(strong_rows),
        "coverage_percent": pct(len(strong_rows) / subset_n) if subset_n else 0.0,
        **metrics,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.rows_jsonl))
    output_dir = Path(args.output_dir)
    neck = build_neck_proxy(rows)
    fat = build_fat_proxy(rows)
    write_json_atomic(neck, output_dir / "neck_proxy.json")
    write_json_atomic(fat, output_dir / "face_and_neck_fat_proxy.json")
    write_csv([neck, fat], output_dir / "proxy_summary.csv", SUMMARY_FIELDS)
    print(f"Wrote proxy outputs to {output_dir}")


if __name__ == "__main__":
    main()
