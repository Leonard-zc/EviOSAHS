from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from eviosahs.eval_four_class import (
    SEVERITY_LABELS,
    compute_four_class_metrics,
    extract_final_severity,
    numeric_to_severity,
    write_confusion_csv,
)
from eviosahs.models import (
    ProgressTracker,
    QwenReasonerRunner,
    append_jsonl,
    build_device_map,
    chunked,
    enforce_output_policy,
    ensure_batch_alignment,
    load_existing_stage_records,
    now_iso,
    read_jsonl,
    to_error_record,
    write_csv,
    write_json_atomic,
    write_run_manifest,
    write_stage_state,
)
from eviosahs.prompts import format_evidence_cards


SEVERITY_SYSTEM_PROMPT = """You are estimating four-class OSAHS screening severity from structured evidence.
This is a screening-oriented severity label, not a definitive clinical diagnosis.
Use both the evidence cards and the clinical summary.
Do not default to the majority class.
Treat normal, mild, moderate, and severe as the only valid labels.
Do not use markdown or code fences."""

SEVERITY_CSV_FIELDS = [
    "image_path",
    "gold_severity",
    "predicted_severity",
    "match",
    "ordinal_error",
    "parse_status",
    "final_response",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four-class severity final stage from an existing F5 reason.jsonl.")
    parser.add_argument("--reason-jsonl", required=True, help="Existing F5 reason.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Output directory for four-class final predictions.")
    parser.add_argument("--reasoner-model-path", default=None, help="Local path or HF id for the text reasoner.")
    parser.add_argument("--device", default=None, help="Reasoner device, for example cuda:1.")
    parser.add_argument("--final-batch-size", type=int, default=8, help="Batch size for severity final prompts.")
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2_calibrated", "v3_rule_assisted"],
        default="v1",
        help="Severity prompt version. v1 preserves the original prompt; v2_calibrated adds an ordinal decision rubric; v3_rule_assisted anchors the LLM to a deterministic clinical/evidence rubric.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for dry runs.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing final_severity outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing outputs and rerun.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after per-sample errors.")
    return parser.parse_args()


def build_paths(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "final_jsonl": str(root / "final_severity.jsonl"),
        "final_csv": str(root / "final_severity.csv"),
        "final_state": str(root / "final_severity.state.json"),
        "final_errors": str(root / "final_severity.errors.jsonl"),
        "metrics_json": str(root / "metrics.json"),
        "confusion_csv": str(root / "confusion_matrix.csv"),
        "run_manifest": str(root / "run.json"),
    }


def build_four_class_prompt_v1(record: dict[str, Any]) -> str:
    evidence_cards_text = format_evidence_cards(record.get("session", []), include_evidence_strength=True)
    clinical_summary = record.get("clinical_summary") or record.get("semantic_text", "")
    return f"""This is a four-class OSAHS screening severity task.
Choose exactly one severity label from: normal, mild, moderate, severe.

Definitions:
- normal: no OSAHS / screening-negative.
- mild: mild OSAHS severity.
- moderate: moderate OSAHS severity.
- severe: severe OSAHS severity.

Evidence cards:
{evidence_cards_text}

Clinical summary:
{clinical_summary}

Decision rules:
1. First assess whether the evidence and clinical profile are compatible with normal, mild, moderate, or severe screening severity.
2. Use the ordered nature of the labels: normal < mild < moderate < severe.
3. Do not collapse all positive cases into severe.
4. Do not default to normal only because polysomnography is unavailable.
5. If adjacent severities are close, choose the lower severity only when the evidence is weak or mixed.
6. Return only the requested fields.

Return exactly these 2 lines:
SeverityReasoning: <1 to 3 concise sentences>
Final severity: <normal|mild|moderate|severe>"""


def build_four_class_prompt_v2_calibrated(record: dict[str, Any]) -> str:
    evidence_cards_text = format_evidence_cards(record.get("session", []), include_evidence_strength=True)
    clinical_summary = record.get("clinical_summary") or record.get("semantic_text", "")
    return f"""This is a four-class OSAHS screening severity task.
Choose exactly one ordered severity label from: normal, mild, moderate, severe.

Important:
- This is not a definitive PSG diagnosis.
- Do not reserve severe for cases with definitive sleep-test evidence.
- Do not use mild as the default positive class.
- Use all four labels when the evidence and clinical burden warrant them.

Evidence cards:
{evidence_cards_text}

Clinical summary:
{clinical_summary}

Calibrated ordinal rubric:
1. First count the evidence cards as supports / against / uncertain and note whether support is weak, moderate, or strong.
2. Then judge clinical burden:
   - low: healthy weight, normal neck size, no major comorbidity, and few supporting anatomy findings.
   - medium: overweight, borderline/elevated neck or waist-hip profile, or several weak anatomical supports.
   - high: obesity, clearly large neck/central adiposity, multiple metabolic/cardiovascular comorbidities, or high clinical burden plus multiple anatomical supports.
3. Map to severity:
   - normal: low clinical burden and evidence mostly against/uncertain.
   - mild: limited positive evidence with low-to-medium clinical burden.
   - moderate: medium clinical burden, obesity without multiple comorbidities, or several supporting anatomical findings.
   - severe: high clinical burden plus multiple supporting findings, obesity with comorbidity, clearly high neck/adiposity burden, or evidence pattern strongly above moderate.
4. If a patient is clinically high-risk, do not downgrade to mild only because individual visual supports are weak.
5. If adjacent classes are close, choose the class that best matches the overall clinical burden rather than always choosing the lower class.

Return exactly these 3 lines:
SeverityEvidenceLevel: <normal_like|mild_like|moderate_like|severe_like>
SeverityReasoning: <1 to 3 concise sentences>
Final severity: <normal|mild|moderate|severe>"""


def _extract_float(text: str, field: str) -> float | None:
    match = re.search(rf"\b{re.escape(field)}\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _extract_text_value(text: str, field: str) -> str:
    match = re.search(rf"\b{re.escape(field)}\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    return match.group(1).strip().lower() if match else ""


def _extract_named_section(text: str, section_name: str) -> str:
    pattern = rf"{re.escape(section_name)}:\s*(.*?)(?=\n[A-Za-z][A-Za-z]+:|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _neck_thresholds(sex: str) -> tuple[float, float]:
    normalized = sex.strip().lower()
    if normalized == "female":
        return 38.0, 40.0
    if normalized == "male":
        return 43.0, 45.0
    return 42.0, 45.0


def _clinical_burden_points(clinical_summary: str) -> dict[str, Any]:
    sex = _extract_text_value(clinical_summary, "Sex")
    bmi = _extract_float(clinical_summary, "BMI")
    neck = _extract_float(clinical_summary, "NeckCircumferenceCm")
    whr_category = _extract_text_value(clinical_summary, "WaistHipRatioCategory")
    bmi_category = _extract_text_value(clinical_summary, "BMICategory")
    positive_risk = _extract_named_section(clinical_summary, "PositiveClinicalRisk").lower()

    bmi_points = 0
    if bmi is not None:
        if bmi >= 35:
            bmi_points = 3
        elif bmi >= 30:
            bmi_points = 2
        elif bmi >= 25:
            bmi_points = 1
    elif "morbid obesity" in bmi_category:
        bmi_points = 3
    elif "obesity" in bmi_category:
        bmi_points = 2
    elif "overweight" in bmi_category:
        bmi_points = 1

    neck_points = 0
    large_neck, very_large_neck = _neck_thresholds(sex)
    if neck is not None:
        if neck >= very_large_neck:
            neck_points = 2
        elif neck >= large_neck:
            neck_points = 1

    whr_points = 0
    if "markedly elevated" in whr_category:
        whr_points = 2
    elif "elevated" in whr_category:
        whr_points = 1
    elif "borderline" in whr_category:
        whr_points = 0.5

    comorbidities = [
        name
        for name in ["hypertension", "diabetes", "heart disease", "hyperlipidemia"]
        if f"history of {name}" in positive_risk
    ]
    comorbidity_points = min(len(comorbidities), 3)
    total_points = bmi_points + neck_points + whr_points + comorbidity_points

    return {
        "sex": sex or "unknown",
        "bmi": bmi,
        "bmi_category": bmi_category or "unknown",
        "bmi_points": bmi_points,
        "neck_circumference_cm": neck,
        "neck_points": neck_points,
        "whr_category": whr_category or "unknown",
        "whr_points": whr_points,
        "comorbidities": comorbidities,
        "comorbidity_points": comorbidity_points,
        "clinical_burden_points": total_points,
    }


def _evidence_counts(record: dict[str, Any]) -> dict[str, int]:
    counts = {
        "supports": 0,
        "against": 0,
        "uncertain": 0,
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
    }
    for item in record.get("session", []):
        card = item.get("evidence_card", {})
        risk_direction = str(card.get("risk_direction", "")).strip().lower()
        confidence = str(card.get("confidence", "")).strip().lower()
        if risk_direction in {"supports", "against", "uncertain"}:
            counts[risk_direction] += 1
        if confidence in {"high", "medium", "low"}:
            counts[f"{confidence}_confidence"] += 1
    return counts


def build_severity_rule_summary(record: dict[str, Any]) -> dict[str, Any]:
    clinical_summary = record.get("clinical_summary") or record.get("semantic_text", "")
    clinical = _clinical_burden_points(clinical_summary)
    evidence = _evidence_counts(record)
    points = float(clinical["clinical_burden_points"])
    supports = evidence["supports"]
    against = evidence["against"]
    uncertain = evidence["uncertain"]

    suggested = "mild"
    rationale: list[str] = []
    if (
        points <= 1
        and clinical["bmi_points"] == 0
        and clinical["neck_points"] == 0
        and clinical["comorbidity_points"] == 0
        and supports <= 5
        and against + uncertain >= 2
    ):
        suggested = "normal"
        rationale.append("low clinical burden anchors normal despite weak visual support")
    elif points >= 6 or (
        points >= 5
        and supports >= 5
    ) or (
        clinical["bmi"] is not None
        and clinical["bmi"] >= 35
        and clinical["neck_points"] >= 1
    ) or (
        clinical["bmi"] is not None
        and clinical["bmi"] >= 30
        and clinical["neck_points"] >= 1
        and clinical["comorbidity_points"] >= 1
    ):
        suggested = "severe"
        rationale.append("high clinical burden crosses severe rule threshold")
    elif points >= 3.5 or supports >= 6 or (
        clinical["bmi"] is not None
        and clinical["bmi"] >= 30
    ):
        suggested = "moderate"
        rationale.append("medium-to-high clinical burden or broad visual support")
    else:
        suggested = "mild"
        rationale.append("positive but not enough clinical burden for moderate/severe")

    if not rationale:
        rationale.append("rule threshold default")

    return {
        **clinical,
        **evidence,
        "rule_suggested_severity": suggested,
        "rule_rationale": "; ".join(rationale),
    }


def format_severity_rule_summary(summary: dict[str, Any]) -> str:
    comorbidities = ", ".join(summary["comorbidities"]) if summary["comorbidities"] else "none"
    return "\n".join(
        [
            f"ClinicalBurdenPoints: {summary['clinical_burden_points']}",
            f"BMI: {summary['bmi']} ({summary['bmi_category']}), bmi_points={summary['bmi_points']}",
            f"NeckCircumferenceCm: {summary['neck_circumference_cm']}, neck_points={summary['neck_points']}",
            f"WaistHipRatioCategory: {summary['whr_category']}, whr_points={summary['whr_points']}",
            f"PositiveComorbidities: {comorbidities}, comorbidity_points={summary['comorbidity_points']}",
            f"EvidenceCounts: supports={summary['supports']}, against={summary['against']}, uncertain={summary['uncertain']}",
            f"EvidenceConfidenceCounts: high={summary['high_confidence']}, medium={summary['medium_confidence']}, low={summary['low_confidence']}",
            f"RuleSuggestedSeverity: {summary['rule_suggested_severity']}",
            f"RuleRationale: {summary['rule_rationale']}",
        ]
    )


def build_four_class_prompt_v3_rule_assisted(record: dict[str, Any]) -> str:
    evidence_cards_text = format_evidence_cards(record.get("session", []), include_evidence_strength=True)
    clinical_summary = record.get("clinical_summary") or record.get("semantic_text", "")
    rule_summary = build_severity_rule_summary(record)
    return f"""This is a four-class OSAHS screening severity task.
Choose exactly one ordered severity label from: normal, mild, moderate, severe.

This v3 prompt is rule-assisted. Use the deterministic rule summary below as the primary anchor.
You may adjust the rule suggestion by at most one adjacent class only when the evidence cards clearly contradict it.
Do not ignore RuleSuggestedSeverity.
Do not avoid severe because this is not a definitive PSG diagnosis.
Do not force all positive cases into mild.
Do not force all low-risk cases into mild; normal remains a valid output.

Rule-assisted summary:
{format_severity_rule_summary(rule_summary)}

Evidence cards:
{evidence_cards_text}

Clinical summary:
{clinical_summary}

Decision contract:
1. Start from RuleSuggestedSeverity.
2. Keep normal when clinical burden is low and visual evidence is mostly against/uncertain.
3. Keep severe when clinical burden is high, especially obesity plus large neck, elevated waist-hip profile, or comorbidity.
4. Use moderate for clear positive burden that does not reach the severe rule.
5. Use mild only for limited positive evidence with low-to-medium clinical burden.
6. If the evidence cards are all weak, do not automatically downgrade severe clinical burden to mild; weak visual evidence is not evidence against clinical burden.

Return exactly these 4 lines:
RuleSuggestedSeverity: <normal|mild|moderate|severe>
SeverityAdjustment: <keep|downshift_one|upshift_one>
SeverityReasoning: <1 to 3 concise sentences>
Final severity: <normal|mild|moderate|severe>"""


def build_four_class_prompt(record: dict[str, Any], prompt_version: str) -> str:
    if prompt_version == "v1":
        return build_four_class_prompt_v1(record)
    if prompt_version == "v2_calibrated":
        return build_four_class_prompt_v2_calibrated(record)
    if prompt_version == "v3_rule_assisted":
        return build_four_class_prompt_v3_rule_assisted(record)
    raise ValueError(f"Unsupported prompt version: {prompt_version}")


def ordinal_error(gold: str, predicted: str) -> int | None:
    if gold not in SEVERITY_LABELS or predicted not in SEVERITY_LABELS:
        return None
    return abs(SEVERITY_LABELS.index(gold) - SEVERITY_LABELS.index(predicted))


def build_final_record(record: dict[str, Any], prompt: str, response: str, prompt_version: str) -> dict[str, Any]:
    gold_severity = numeric_to_severity(record["numeric_label"])
    predicted_severity, parse_status = extract_final_severity(response)
    return {
        "sample_index": record.get("sample_index"),
        "image_path": record["image_path"],
        "semantic_text": record.get("semantic_text", ""),
        "clinical_summary": record.get("clinical_summary", ""),
        "numeric_label": int(record["numeric_label"]),
        "gold_severity": gold_severity,
        "prompt_version": prompt_version,
        "severity_rule_summary": build_severity_rule_summary(record) if prompt_version == "v3_rule_assisted" else {},
        "severity_prompt": prompt,
        "final_response": response.strip(),
        "predicted_severity": predicted_severity,
        "parse_status": parse_status,
        "match": predicted_severity == gold_severity,
        "ordinal_error": ordinal_error(gold_severity, predicted_severity),
        "saved_at": now_iso(),
    }


def write_final_csv(records: list[dict[str, Any]], csv_path: str | Path) -> None:
    rows = []
    for record in sorted(records, key=lambda item: int(item.get("sample_index", 10**9))):
        rows.append(
            {
                "image_path": record["image_path"],
                "gold_severity": record["gold_severity"],
                "predicted_severity": record.get("predicted_severity", "unknown"),
                "match": record.get("match", False),
                "ordinal_error": "" if record.get("ordinal_error") is None else record.get("ordinal_error"),
                "parse_status": record.get("parse_status", "unparsed"),
                "final_response": record.get("final_response", ""),
            }
        )
    write_csv(rows, csv_path, fieldnames=SEVERITY_CSV_FIELDS)


def finalize_outputs(final_records: list[dict[str, Any]], paths: dict[str, str]) -> None:
    ordered_records = sorted(final_records, key=lambda item: int(item.get("sample_index", 10**9)))
    metrics = compute_four_class_metrics(ordered_records)
    write_json_atomic({"pred_file": paths["final_jsonl"], **metrics}, paths["metrics_json"])
    write_confusion_csv(metrics, paths["confusion_csv"])
    write_final_csv(ordered_records, paths["final_csv"])


def main() -> None:
    args = parse_args()
    paths = build_paths(args.output_dir)
    enforce_output_policy(paths, args.resume, args.overwrite)

    reason_records = read_jsonl(args.reason_jsonl, strict=False)
    if args.limit is not None:
        reason_records = reason_records[: args.limit]
    target_paths = {record["image_path"] for record in reason_records}
    existing = load_existing_stage_records(paths["final_jsonl"], target_paths, args.resume)
    final_lookup = {record["image_path"]: record for record in existing}
    if existing:
        write_final_csv(existing, paths["final_csv"])

    runner = QwenReasonerRunner(model_name=args.reasoner_model_path, device_map=build_device_map(args.device))
    write_run_manifest(
        paths["run_manifest"],
        script_name=Path(__file__).name,
        args=vars(args),
        resolved_models={"reasoner_model": runner.model_name, "device": args.device or "auto"},
        output_dir=args.output_dir,
    )

    tracker = ProgressTracker("final_severity", total=len(reason_records), completed=len(existing))
    write_stage_state(
        paths["final_state"],
        tracker,
        status="running",
        output_path=paths["final_jsonl"],
        error_path=paths["final_errors"],
        extra={"model_path": runner.model_name, "final_batch_size": args.final_batch_size, "prompt_version": args.prompt_version},
    )

    pending = [record for record in reason_records if record["image_path"] not in final_lookup]
    try:
        for batch in chunked(pending, args.final_batch_size):
            prompts = [build_four_class_prompt(record, args.prompt_version) for record in batch]
            try:
                responses = runner.generate_batch(prompts, system_prompt=SEVERITY_SYSTEM_PROMPT)
            except Exception:
                if len(batch) == 1:
                    raise
                responses = [
                    runner.generate(prompt, system_prompt=SEVERITY_SYSTEM_PROMPT)
                    for prompt in prompts
                ]
            ensure_batch_alignment(responses, len(batch), "four_class_final_from_reason")

            for record, prompt, response in zip(batch, prompts, responses):
                image_name = record["image_path"]
                try:
                    final_record = build_final_record(record, prompt, response, args.prompt_version)
                    append_jsonl(final_record, paths["final_jsonl"])
                    final_lookup[image_name] = final_record
                    tracker.advance(image_name)
                    write_stage_state(
                        paths["final_state"],
                        tracker,
                        status="running",
                        output_path=paths["final_jsonl"],
                        error_path=paths["final_errors"],
                        current_item=image_name,
                        extra={"model_path": runner.model_name, "final_batch_size": args.final_batch_size, "prompt_version": args.prompt_version},
                    )
                except Exception as exc:
                    append_jsonl(to_error_record("final_severity", image_name, exc), paths["final_errors"])
                    tracker.mark_error(image_name)
                    if not args.continue_on_error:
                        raise
    finally:
        runner.close()

    final_records = sorted(final_lookup.values(), key=lambda item: int(item.get("sample_index", 10**9)))
    finalize_outputs(final_records, paths)
    write_stage_state(
        paths["final_state"],
        tracker,
        status="completed_with_errors" if tracker.errors else "completed",
        output_path=paths["final_jsonl"],
        error_path=paths["final_errors"],
        extra={"model_path": runner.model_name, "final_batch_size": args.final_batch_size, "prompt_version": args.prompt_version},
    )


if __name__ == "__main__":
    main()
