from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from eviosahs.models import (
    VISUAL_NOISE_KEYWORDS,
    compute_binary_metrics,
    load_prediction_records,
    now_iso,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EviOSAHS final prediction files.")
    parser.add_argument("--pred-file", required=True, help="Path to final.jsonl or final.csv")
    parser.add_argument("--analysis-out", required=True, help="Path to write analysis.json")
    return parser.parse_args()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _sample_summary(record: dict[str, Any]) -> dict[str, Any]:
    session = record.get("session", [])
    evidence = []
    for item in session[:3]:
        evidence_card = item.get("evidence_card", {})
        evidence.append(
            {
                "session_index": item.get("session_index"),
                "anatomy_target": item.get("anatomy_target"),
                "visibility": item.get("visibility"),
                "visual_observation": item.get("visual_observation"),
                "risk_direction": evidence_card.get("risk_direction"),
                "evidence_strength": evidence_card.get("evidence_strength"),
                "confidence": evidence_card.get("confidence"),
                "evidence_summary": evidence_card.get("evidence_summary"),
            }
        )
    return {
        "image_path": record.get("image_path"),
        "gold_label": record.get("binary_label"),
        "predicted_label": record.get("predicted_label"),
        "parse_status": record.get("parse_status"),
        "final_response_excerpt": _safe_text(record.get("final_response"))[:400],
        "evidence_preview": evidence,
    }


def build_analysis_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_binary_metrics(records)
    predicted_distribution = Counter()
    visual_parse = Counter()
    evidence_parse = Counter()
    noise_counter = Counter()
    reasoning_lengths: list[int] = []
    final_prompt_lengths: list[int] = []
    final_response_lengths: list[int] = []
    fn_samples: list[dict[str, Any]] = []
    fp_samples: list[dict[str, Any]] = []
    unknown_samples: list[dict[str, Any]] = []

    for record in records:
        predicted_distribution[_safe_text(record.get("predicted_label", "unknown")).lower()] += 1
        reasoning_lengths.append(len(_safe_text(record.get("reasoning_summary"))))
        final_prompt_lengths.append(len(_safe_text(record.get("final_prompt"))))
        final_response_lengths.append(len(_safe_text(record.get("final_response")) or _safe_text(record.get("raw_response"))))

        session = record.get("session", [])
        for item in session:
            visual_parse[_safe_text(item.get("visual_parse_status", "missing"))] += 1
            evidence_parse[_safe_text(item.get("evidence_parse_status", "missing"))] += 1
            visual_text = " ".join(
                [
                    _safe_text(item.get("raw_visual_response")),
                    _safe_text(item.get("visual_observation")),
                ]
            ).lower()
            for keyword in VISUAL_NOISE_KEYWORDS:
                if keyword in visual_text:
                    noise_counter[keyword] += 1

        gold = record.get("binary_label")
        predicted = record.get("predicted_label")
        if gold == "yes" and predicted != "yes":
            fn_samples.append(_sample_summary(record))
        elif gold == "no" and predicted == "yes":
            fp_samples.append(_sample_summary(record))
        if predicted not in {"yes", "no"}:
            unknown_samples.append(_sample_summary(record))

    def stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(mean(values), 2),
            "median": float(median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    return {
        "created_at": now_iso(),
        "metrics": metrics,
        "prediction_distribution": dict(predicted_distribution),
        "predicted_distribution": dict(predicted_distribution),
        "visual_parse_status": dict(visual_parse),
        "evidence_parse_status": dict(evidence_parse),
        "visual_noise_keyword_counts": dict(noise_counter),
        "reasoning_summary_length": stats(reasoning_lengths),
        "final_prompt_length": stats(final_prompt_lengths),
        "final_response_length": stats(final_response_lengths),
        "fn_samples": fn_samples,
        "fp_samples": fp_samples,
        "unknown_samples": unknown_samples,
    }


def main() -> None:
    args = parse_args()
    records = load_prediction_records(args.pred_file)
    payload = build_analysis_payload(records)
    write_json_atomic(payload, args.analysis_out)


if __name__ == "__main__":
    main()
