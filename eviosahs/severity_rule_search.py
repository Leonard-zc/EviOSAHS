from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

from eviosahs.eval_four_class import compute_four_class_metrics, numeric_to_severity, write_confusion_csv
from eviosahs.four_class_final_from_reason import _clinical_burden_points, _evidence_counts, ordinal_error
from eviosahs.models import now_iso, read_jsonl, write_json_atomic


LABELS = ["normal", "mild", "moderate", "severe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline rule-threshold search for four-class OSAHS severity.")
    parser.add_argument("--reason-jsonl", required=True, help="F5 reason.jsonl with clinical_summary and evidence cards.")
    parser.add_argument("--output-dir", default="outputs/severity/Q4_6_rule_threshold_search")
    parser.add_argument("--top-k", type=int, default=25, help="Number of top configs to write.")
    parser.add_argument(
        "--selection-metric",
        choices=["composite", "macro_f1", "qwk", "balanced_accuracy", "mae"],
        default="composite",
        help="Metric used to select the best threshold set.",
    )
    return parser.parse_args()


def record_components(record: dict[str, Any]) -> dict[str, Any]:
    clinical_summary = record.get("clinical_summary") or record.get("semantic_text", "")
    clinical = _clinical_burden_points(clinical_summary)
    evidence = _evidence_counts(record)
    return {
        **clinical,
        **evidence,
    }


def severity_score(components: dict[str, Any], params: dict[str, float]) -> float:
    return (
        params["bmi_weight"] * float(components["bmi_points"])
        + params["neck_weight"] * float(components["neck_points"])
        + params["whr_weight"] * float(components["whr_points"])
        + params["comorbidity_weight"] * float(components["comorbidity_points"])
        + params["support_weight"] * float(components["supports"])
        - params["against_weight"] * float(components["against"])
        - params["uncertain_weight"] * float(components["uncertain"])
    )


def score_to_label(score: float, thresholds: dict[str, float]) -> str:
    if score <= thresholds["normal_max"]:
        return "normal"
    if score <= thresholds["mild_max"]:
        return "mild"
    if score <= thresholds["moderate_max"]:
        return "moderate"
    return "severe"


def build_prediction_records(
    records: list[dict[str, Any]],
    components_by_image: dict[str, dict[str, Any]],
    params: dict[str, float],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        components = components_by_image[record["image_path"]]
        score = severity_score(components, params)
        predicted = score_to_label(score, thresholds)
        gold = numeric_to_severity(record["numeric_label"])
        predictions.append(
            {
                "sample_index": record.get("sample_index"),
                "image_path": record["image_path"],
                "numeric_label": int(record["numeric_label"]),
                "gold_severity": gold,
                "predicted_severity": predicted,
                "match": predicted == gold,
                "ordinal_error": ordinal_error(gold, predicted),
                "parse_status": "rule_only",
                "severity_score": round(score, 6),
                "rule_params": {**params, **thresholds},
                "rule_components": components,
                "final_response": f"Rule-only severity score={score:.4f}; final severity={predicted}",
                "saved_at": now_iso(),
            }
        )
    return predictions


def metric_value(metrics: dict[str, Any], selection_metric: str) -> float:
    if selection_metric == "macro_f1":
        return float(metrics["macro_f1"])
    if selection_metric == "qwk":
        return float(metrics["quadratic_weighted_kappa_parsed"])
    if selection_metric == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    if selection_metric == "mae":
        return -float(metrics["mae_parsed"])

    # Composite keeps the search from overfitting to a single display metric.
    return (
        float(metrics["macro_f1"])
        + 0.35 * float(metrics["balanced_accuracy"])
        + 0.35 * float(metrics["quadratic_weighted_kappa_parsed"])
        - 0.10 * float(metrics["mae_parsed"])
    )


def has_all_classes(metrics: dict[str, Any]) -> bool:
    predicted_counts = metrics["predicted_counts"]
    return all(predicted_counts.get(label, 0) > 0 for label in LABELS)


def search_configs(
    records: list[dict[str, Any]],
    components_by_image: dict[str, dict[str, Any]],
    selection_metric: str,
) -> list[dict[str, Any]]:
    weight_grid = {
        "bmi_weight": [1.0, 1.2],
        "neck_weight": [0.8, 1.0],
        "whr_weight": [0.6, 0.8],
        "comorbidity_weight": [0.8, 1.0],
        "support_weight": [0.0, 0.1, 0.2],
        "against_weight": [0.0, 0.1],
        "uncertain_weight": [0.0, 0.05],
    }
    threshold_grid = {
        "normal_max": [0.8, 1.0, 1.2, 1.5],
        "mild_max": [2.0, 2.4, 2.8, 3.2],
        "moderate_max": [3.6, 4.0, 4.4, 4.8, 5.2],
    }

    results: list[dict[str, Any]] = []
    weight_keys = list(weight_grid)
    threshold_keys = list(threshold_grid)
    for weight_values in itertools.product(*(weight_grid[key] for key in weight_keys)):
        params = dict(zip(weight_keys, weight_values))
        for threshold_values in itertools.product(*(threshold_grid[key] for key in threshold_keys)):
            thresholds = dict(zip(threshold_keys, threshold_values))
            if not (thresholds["normal_max"] < thresholds["mild_max"] < thresholds["moderate_max"]):
                continue
            predictions = build_prediction_records(records, components_by_image, params, thresholds)
            metrics = compute_four_class_metrics(predictions)
            if not has_all_classes(metrics):
                continue
            score = metric_value(metrics, selection_metric)
            results.append(
                {
                    "selection_score": score,
                    "params": params,
                    "thresholds": thresholds,
                    "metrics": metrics,
                }
            )
    return sorted(results, key=lambda item: item["selection_score"], reverse=True)


def flatten_result(index: int, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    report = metrics["report"]
    counts = metrics["predicted_counts"]
    row = {
        "rank": index,
        "selection_score": round(float(result["selection_score"]), 8),
        **result["params"],
        **result["thresholds"],
        "accuracy_percent": report["Accuracy (%)"],
        "macro_f1_percent": report["Macro-F1 (%)"],
        "weighted_f1_percent": report["Weighted-F1 (%)"],
        "balanced_accuracy_percent": report["Balanced Accuracy (%)"],
        "qwk": report["Quadratic Weighted Kappa (parsed)"],
        "mae": report["MAE (parsed)"],
    }
    for label in LABELS:
        row[f"pred_{label}"] = counts.get(label, 0)
        row[f"recall_{label}_percent"] = metrics["per_class"][label]["recall_percent"]
        row[f"f1_{label}_percent"] = metrics["per_class"][label]["f1_percent"]
    return row


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.reason_jsonl, strict=False)
    components_by_image = {record["image_path"]: record_components(record) for record in records}

    results = search_configs(records, components_by_image, args.selection_metric)
    if not results:
        raise RuntimeError("No non-collapsed rule configuration found.")

    top_results = results[: args.top_k]
    top_rows = [flatten_result(index + 1, result) for index, result in enumerate(top_results)]
    write_csv_rows(output_dir / "rule_search_top.csv", top_rows)

    best = top_results[0]
    best_predictions = build_prediction_records(records, components_by_image, best["params"], best["thresholds"])
    best_metrics = compute_four_class_metrics(best_predictions)
    write_jsonl(output_dir / "best_rule_predictions.jsonl", best_predictions)
    write_json_atomic(
        {
            "created_at": now_iso(),
            "reason_jsonl": args.reason_jsonl,
            "selection_metric": args.selection_metric,
            "selection_score": best["selection_score"],
            "params": best["params"],
            "thresholds": best["thresholds"],
            **best_metrics,
        },
        output_dir / "best_rule_metrics.json",
    )
    write_confusion_csv(best_metrics, output_dir / "best_rule_confusion_matrix.csv")

    print(f"Searched {len(results)} non-collapsed configs.")
    print(f"Wrote {output_dir / 'rule_search_top.csv'}")
    print(f"Wrote {output_dir / 'best_rule_metrics.json'}")
    print(json.dumps(best_metrics["report"], indent=2, ensure_ascii=False))
    print(f"Predicted counts: {best_metrics['predicted_counts']}")


if __name__ == "__main__":
    main()
