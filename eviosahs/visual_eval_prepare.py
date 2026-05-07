from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from eviosahs.models import read_jsonl, write_csv, write_jsonl


CSV_FIELDS = [
    "sample_index",
    "image_path",
    "numeric_label",
    "binary_label",
    "sex",
    "neck_circumference_cm",
    "bmi",
    "bmi_category",
    "waist_hip_ratio",
    "waist_hip_ratio_category",
    "session_index",
    "anatomy_target",
    "visual_observation",
    "visibility",
    "visual_parse_status",
    "raw_visual_response",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten F5 visual.jsonl into question-level rows.")
    parser.add_argument("--visual-jsonl", required=True, help="Input visual.jsonl from a two-stage run")
    parser.add_argument("--output-jsonl", required=True, help="Output question-level JSONL rows")
    parser.add_argument("--output-csv", required=True, help="Output question-level CSV rows")
    return parser.parse_args()


def _field(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _float_field(pattern: str, text: str) -> float | None:
    raw = _field(pattern, text)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_clinical_summary(summary: str) -> dict[str, Any]:
    return {
        "sex": _field(r"Sex:\s*([^\n]+)", summary).lower(),
        "neck_circumference_cm": _float_field(r"NeckCircumferenceCm:\s*([0-9.]+)", summary),
        "bmi": _float_field(r"BMI:\s*([0-9.]+)", summary),
        "bmi_category": _field(r"BMICategory:\s*([^\n]+)", summary).lower(),
        "waist_hip_ratio": _float_field(r"WaistHipRatio:\s*([0-9.]+)", summary),
        "waist_hip_ratio_category": _field(r"WaistHipRatioCategory:\s*([^\n]+)", summary).lower(),
    }


def build_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        clinical = parse_clinical_summary(str(record.get("clinical_summary", "")))
        base = {
            "sample_index": record.get("sample_index"),
            "image_path": record.get("image_path"),
            "numeric_label": record.get("numeric_label"),
            "binary_label": record.get("binary_label"),
            **clinical,
        }
        for session in record.get("session", []):
            row = {
                **base,
                "session_index": session.get("session_index"),
                "anatomy_target": session.get("anatomy_target"),
                "visual_observation": session.get("visual_observation"),
                "visibility": str(session.get("visibility", "")).lower(),
                "visual_parse_status": str(session.get("visual_parse_status", "")).lower(),
                "raw_visual_response": session.get("raw_visual_response"),
            }
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    rows = build_rows(read_jsonl(Path(args.visual_jsonl)))
    write_jsonl(rows, args.output_jsonl)
    write_csv(rows, args.output_csv, CSV_FIELDS)
    print(f"Wrote {len(rows)} rows to {args.output_jsonl} and {args.output_csv}")


if __name__ == "__main__":
    main()
