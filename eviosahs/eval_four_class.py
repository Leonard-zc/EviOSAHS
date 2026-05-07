from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SEVERITY_LABELS = ["normal", "mild", "moderate", "severe"]
SEVERITY_TO_INDEX = {label: index for index, label in enumerate(SEVERITY_LABELS)}


def numeric_to_severity(label: Any) -> str:
    value = int(float(str(label).strip()))
    if value == 0:
        return "normal"
    if value == 1:
        return "mild"
    if value == 2:
        return "moderate"
    if value == 3:
        return "severe"
    raise ValueError(f"Unsupported severity label: {label}")


def normalize_severity_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    if not text:
        return "unknown"
    if re.fullmatch(r"[0-3]", text):
        return numeric_to_severity(text)
    if "severe" in text:
        return "severe"
    if "moderate" in text:
        return "moderate"
    if "mild" in text:
        return "mild"
    if any(token in text for token in ["normal", "no osa", "no osahs", "no osa", "non osa", "negative", "none"]):
        return "normal"
    return "unknown"


def extract_final_severity(text: str) -> tuple[str, str]:
    final_match = re.search(
        r"final\s+severity\s*:\s*(normal|no[_\-\s]?osa|no[_\-\s]?osahs|mild|moderate|severe|[0-3])\b",
        text,
        flags=re.IGNORECASE,
    )
    if final_match:
        return normalize_severity_label(final_match.group(1)), "final_severity"

    severity_matches = re.findall(
        r"\b(normal|no[_\-\s]?osa|no[_\-\s]?osahs|mild|moderate|severe)\b",
        text,
        flags=re.IGNORECASE,
    )
    if severity_matches:
        return normalize_severity_label(severity_matches[-1]), "fallback_last_label"

    numeric_matches = re.findall(r"\b([0-3])\b", text)
    if numeric_matches:
        return numeric_to_severity(numeric_matches[-1]), "fallback_last_numeric"

    return "unknown", "unparsed"


def read_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    pred_path = Path(path)
    if pred_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with pred_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records
    if pred_path.suffix.lower() == ".csv":
        with pred_path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported prediction file format: {pred_path}")


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _quadratic_weighted_kappa(y_true: list[int], y_pred: list[int], num_classes: int) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0

    observed = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    true_hist = [0.0 for _ in range(num_classes)]
    pred_hist = [0.0 for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        observed[truth][pred] += 1.0
        true_hist[truth] += 1.0
        pred_hist[pred] += 1.0

    total = float(len(y_true))
    weighted_observed = 0.0
    weighted_expected = 0.0
    denominator = float((num_classes - 1) ** 2)
    for i in range(num_classes):
        for j in range(num_classes):
            weight = ((i - j) ** 2) / denominator if denominator else 0.0
            expected = true_hist[i] * pred_hist[j] / total
            weighted_observed += weight * observed[i][j]
            weighted_expected += weight * expected

    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else 0.0
    return 1.0 - weighted_observed / weighted_expected


def compute_four_class_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_records = []
    for record in records:
        raw_gold = record.get("gold_severity") or record.get("severity_label")
        if not raw_gold and "numeric_label" in record:
            raw_gold = numeric_to_severity(record["numeric_label"])
        gold = normalize_severity_label(raw_gold)
        if gold in SEVERITY_TO_INDEX:
            valid_records.append((record, gold))

    confusion = [[0 for _ in SEVERITY_LABELS] for _ in SEVERITY_LABELS]
    support = {label: 0 for label in SEVERITY_LABELS}
    predicted_counts = {label: 0 for label in SEVERITY_LABELS}
    correct = 0
    parsed = 0
    unknown = 0
    parsed_true_indices: list[int] = []
    parsed_pred_indices: list[int] = []
    ordinal_errors: list[int] = []

    for record, gold in valid_records:
        predicted = normalize_severity_label(
            record.get("predicted_severity")
            or record.get("predicted_label")
            or record.get("final_severity")
        )
        gold_index = SEVERITY_TO_INDEX[gold]
        support[gold] += 1
        if predicted not in SEVERITY_TO_INDEX:
            unknown += 1
            continue

        pred_index = SEVERITY_TO_INDEX[predicted]
        parsed += 1
        predicted_counts[predicted] += 1
        confusion[gold_index][pred_index] += 1
        parsed_true_indices.append(gold_index)
        parsed_pred_indices.append(pred_index)
        ordinal_errors.append(abs(gold_index - pred_index))
        if gold_index == pred_index:
            correct += 1

    total = len(valid_records)
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    weighted_f1_sum = 0.0
    for label in SEVERITY_LABELS:
        index = SEVERITY_TO_INDEX[label]
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(SEVERITY_LABELS)) if row != index)
        fn = support[label] - tp
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, support[label])
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        recalls.append(recall)
        f1_values.append(f1)
        weighted_f1_sum += f1 * support[label]
        per_class[label] = {
            "support": support[label],
            "predicted": predicted_counts[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_percent": round(precision * 100, 4),
            "recall_percent": round(recall * 100, 4),
            "f1_percent": round(f1 * 100, 4),
        }

    accuracy = _safe_divide(correct, total)
    macro_f1 = _safe_divide(sum(f1_values), len(f1_values))
    weighted_f1 = _safe_divide(weighted_f1_sum, total)
    balanced_accuracy = _safe_divide(sum(recalls), len(recalls))
    qwk = _quadratic_weighted_kappa(parsed_true_indices, parsed_pred_indices, len(SEVERITY_LABELS))
    mae = _safe_divide(sum(ordinal_errors), len(ordinal_errors))

    return {
        "total": total,
        "valid": total,
        "parsed": parsed,
        "unknown": unknown,
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "quadratic_weighted_kappa_parsed": qwk,
        "mae_parsed": mae,
        "support": support,
        "predicted_counts": predicted_counts | {"unknown": unknown},
        "per_class": per_class,
        "confusion_matrix": {
            "labels": SEVERITY_LABELS,
            "matrix": confusion,
        },
        "report": {
            "Accuracy (%)": round(accuracy * 100, 4),
            "Macro-F1 (%)": round(macro_f1 * 100, 4),
            "Weighted-F1 (%)": round(weighted_f1 * 100, 4),
            "Balanced Accuracy (%)": round(balanced_accuracy * 100, 4),
            "Quadratic Weighted Kappa (parsed)": round(qwk, 4),
            "MAE (parsed)": round(mae, 4),
        },
    }


def write_confusion_csv(metrics: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate four-class OSAHS severity predictions.")
    parser.add_argument("--pred-jsonl", required=True, help="Path to final_severity.jsonl or compatible CSV.")
    parser.add_argument("--output-json", required=True, help="Path to write metrics.json.")
    parser.add_argument("--confusion-csv", required=True, help="Path to write 4x4 confusion matrix CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_prediction_records(args.pred_jsonl)
    metrics = compute_four_class_metrics(records)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"pred_file": args.pred_jsonl, **metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_confusion_csv(metrics, args.confusion_csv)


if __name__ == "__main__":
    main()
