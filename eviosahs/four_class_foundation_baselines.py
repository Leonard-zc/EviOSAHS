from __future__ import annotations

import argparse
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
    QwenVLRunner,
    append_jsonl,
    build_clinical_summary_from_structured_text,
    build_device_map,
    chunked,
    enforce_output_policy,
    ensure_batch_alignment,
    load_existing_stage_records,
    load_pickle_dataset,
    now_iso,
    resolve_image_path,
    to_error_record,
    write_csv,
    write_json_atomic,
    write_run_manifest,
    write_stage_state,
)


SEVERITY_SYSTEM_PROMPT = """You are estimating four-class OSAHS screening severity.
This is a screening-oriented severity label, not a definitive clinical diagnosis.
Choose exactly one of: normal, mild, moderate, severe.
Do not default to the majority class.
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
    parser = argparse.ArgumentParser(description="Run four-class foundation baselines for OSAHS severity.")
    parser.add_argument(
        "--method",
        required=True,
        choices=["clinical_only_qwen_text", "qwen_vl_direct"],
        help="Four-class baseline method.",
    )
    parser.add_argument("--input-pkl", required=True, help="Path to the local cohort pickle.")
    parser.add_argument("--image-dir", default=None, help="Directory containing images. Required for qwen_vl_direct.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--qwen-text-model-path", default=None, help="Local path or HF id for Qwen text model.")
    parser.add_argument("--qwen-vl-model-path", default=None, help="Local path or HF id for Qwen-VL model.")
    parser.add_argument("--device", default=None, help="Device, for example cuda:0.")
    parser.add_argument("--final-batch-size", type=int, default=8, help="Batch size for final prompts.")
    parser.add_argument("--clinical-source", choices=["clean", "semantic"], default="clean")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for dry runs.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs.")
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


def resolve_clinical_summary(sample: dict[str, Any], clinical_source: str) -> str:
    if clinical_source == "semantic":
        return sample["semantic_text"]
    rebuilt = build_clinical_summary_from_structured_text(sample.get("text"))
    return rebuilt or sample["semantic_text"]


def build_base_record(sample: dict[str, Any], sample_index: int, args: argparse.Namespace) -> dict[str, Any]:
    image_name = f"{sample['id']}.jpg"
    record = {
        "sample_index": sample_index,
        "image_path": image_name,
        "semantic_text": sample["semantic_text"],
        "clinical_summary": resolve_clinical_summary(sample, args.clinical_source),
        "numeric_label": int(sample["label"]),
        "gold_severity": numeric_to_severity(sample["label"]),
    }
    if args.image_dir:
        record["resolved_image_path"] = str(resolve_image_path(args.image_dir, image_name))
    return record


def build_clinical_only_prompt(clinical_summary: str) -> str:
    return f"""This is a four-class OSAHS screening severity task.
Use only the structured clinical summary.
Choose exactly one severity label from: normal, mild, moderate, severe.

Definitions:
- normal: no OSAHS / screening-negative.
- mild: mild OSAHS severity.
- moderate: moderate OSAHS severity.
- severe: severe OSAHS severity.

Clinical summary:
{clinical_summary}

Decision rules:
1. Use only the clinical information provided.
2. Do not infer unavailable image findings.
3. Use the ordered nature of the labels: normal < mild < moderate < severe.
4. Do not collapse all positive cases into severe.
5. Return only the requested fields.

Return exactly these 2 lines:
SeverityReasoning: <1 to 3 concise sentences>
Final severity: <normal|mild|moderate|severe>"""


def build_direct_multimodal_prompt(clinical_summary: str) -> str:
    return f"""This is a four-class OSAHS screening severity task.
Use the patient image and the structured clinical summary.
Choose exactly one severity label from: normal, mild, moderate, severe.

Definitions:
- normal: no OSAHS / screening-negative.
- mild: mild OSAHS severity.
- moderate: moderate OSAHS severity.
- severe: severe OSAHS severity.

Clinical summary:
{clinical_summary}

Decision rules:
1. Use visible OSAHS-relevant facial and neck anatomy together with the clinical summary.
2. Ignore clothing, background, accessories, and unrelated image content.
3. Use the ordered nature of the labels: normal < mild < moderate < severe.
4. Do not collapse all positive cases into severe.
5. Do not default to normal only because polysomnography is unavailable.
6. Return only the requested fields.

Return exactly these 2 lines:
SeverityReasoning: <1 to 3 concise sentences>
Final severity: <normal|mild|moderate|severe>"""


def ordinal_error(gold: str, predicted: str) -> int | None:
    if gold not in SEVERITY_LABELS or predicted not in SEVERITY_LABELS:
        return None
    return abs(SEVERITY_LABELS.index(gold) - SEVERITY_LABELS.index(predicted))


def build_final_record(base_record: dict[str, Any], prompt: str, response: str, method: str) -> dict[str, Any]:
    predicted_severity, parse_status = extract_final_severity(response)
    gold_severity = base_record["gold_severity"]
    return {
        **base_record,
        "method": method,
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


def run_clinical_only(dataset: list[dict[str, Any]], args: argparse.Namespace, paths: dict[str, str]) -> None:
    runner = QwenReasonerRunner(model_name=args.qwen_text_model_path, device_map=build_device_map(args.device))
    target_paths = {f"{sample['id']}.jpg" for sample in dataset}
    existing = load_existing_stage_records(paths["final_jsonl"], target_paths, args.resume)
    final_lookup = {record["image_path"]: record for record in existing}
    if existing:
        write_final_csv(existing, paths["final_csv"])

    tracker = ProgressTracker("final_severity", total=len(dataset), completed=len(existing))
    write_stage_state(
        paths["final_state"],
        tracker,
        status="running",
        output_path=paths["final_jsonl"],
        error_path=paths["final_errors"],
        extra={"method": args.method, "model_path": runner.model_name, "final_batch_size": args.final_batch_size},
    )

    pending = [
        build_base_record(sample, sample_index, args)
        for sample_index, sample in enumerate(dataset)
        if f"{sample['id']}.jpg" not in final_lookup
    ]
    try:
        for batch in chunked(pending, args.final_batch_size):
            prompts = [build_clinical_only_prompt(record["clinical_summary"]) for record in batch]
            try:
                responses = runner.generate_batch(prompts, system_prompt=SEVERITY_SYSTEM_PROMPT)
            except Exception:
                if len(batch) == 1:
                    raise
                responses = [runner.generate(prompt, system_prompt=SEVERITY_SYSTEM_PROMPT) for prompt in prompts]
            ensure_batch_alignment(responses, len(batch), "four_class_clinical_only")
            for record, prompt, response in zip(batch, prompts, responses):
                image_name = record["image_path"]
                try:
                    final_record = build_final_record(record, prompt, response, args.method)
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
                        extra={"method": args.method, "model_path": runner.model_name, "final_batch_size": args.final_batch_size},
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
        extra={"method": args.method, "model_path": runner.model_name, "final_batch_size": args.final_batch_size},
    )


def run_qwen_vl_direct(dataset: list[dict[str, Any]], args: argparse.Namespace, paths: dict[str, str]) -> None:
    if not args.image_dir:
        raise ValueError("--image-dir is required for qwen_vl_direct.")

    runner = QwenVLRunner(model_name=args.qwen_vl_model_path, device_map=build_device_map(args.device))
    target_paths = {f"{sample['id']}.jpg" for sample in dataset}
    existing = load_existing_stage_records(paths["final_jsonl"], target_paths, args.resume)
    final_lookup = {record["image_path"]: record for record in existing}
    if existing:
        write_final_csv(existing, paths["final_csv"])

    tracker = ProgressTracker("final_severity", total=len(dataset), completed=len(existing))
    write_stage_state(
        paths["final_state"],
        tracker,
        status="running",
        output_path=paths["final_jsonl"],
        error_path=paths["final_errors"],
        extra={"method": args.method, "model_path": runner.model_name, "final_batch_size": args.final_batch_size},
    )

    pending = [
        build_base_record(sample, sample_index, args)
        for sample_index, sample in enumerate(dataset)
        if f"{sample['id']}.jpg" not in final_lookup
    ]
    try:
        for batch in chunked(pending, args.final_batch_size):
            image_paths = [Path(record["resolved_image_path"]) for record in batch]
            prompts = [build_direct_multimodal_prompt(record["clinical_summary"]) for record in batch]
            try:
                responses = runner.generate_batch(image_paths, prompts, system_prompt=SEVERITY_SYSTEM_PROMPT)
            except Exception:
                if len(batch) == 1:
                    raise
                responses = [
                    runner.generate(image_path=image_path, prompt=prompt, system_prompt=SEVERITY_SYSTEM_PROMPT)
                    for image_path, prompt in zip(image_paths, prompts)
                ]
            ensure_batch_alignment(responses, len(batch), "four_class_qwen_vl_direct")
            for record, prompt, response in zip(batch, prompts, responses):
                image_name = record["image_path"]
                try:
                    final_record = build_final_record(record, prompt, response, args.method)
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
                        extra={"method": args.method, "model_path": runner.model_name, "final_batch_size": args.final_batch_size},
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
        extra={"method": args.method, "model_path": runner.model_name, "final_batch_size": args.final_batch_size},
    )


def main() -> None:
    args = parse_args()
    paths = build_paths(args.output_dir)
    enforce_output_policy(paths, args.resume, args.overwrite)

    dataset = load_pickle_dataset(args.input_pkl)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    write_run_manifest(
        paths["run_manifest"],
        script_name=Path(__file__).name,
        args=vars(args),
        resolved_models={
            "qwen_text": args.qwen_text_model_path or "",
            "qwen_vl": args.qwen_vl_model_path or "",
            "device": args.device or "auto",
        },
        output_dir=args.output_dir,
    )

    if args.method == "clinical_only_qwen_text":
        run_clinical_only(dataset, args, paths)
    elif args.method == "qwen_vl_direct":
        run_qwen_vl_direct(dataset, args, paths)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


if __name__ == "__main__":
    main()
