from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path
from typing import Any

from eviosahs.analyze_run import build_analysis_payload
from eviosahs.models import (
    FINAL_CSV_FIELDS,
    Llava16VisualRunner,
    ProgressTracker,
    QwenReasonerRunner,
    QwenVLRunner,
    append_csv_row,
    append_jsonl,
    binary_label_from_numeric,
    build_clinical_summary_from_structured_text,
    build_device_map,
    build_output_paths,
    build_record_lookup,
    build_record_index,
    canonicalize_stage_records,
    chunked,
    compute_binary_metrics,
    enforce_output_policy,
    ensure_batch_alignment,
    extract_binary_answer,
    limit_records,
    load_existing_stage_records,
    load_pickle_dataset,
    now_iso,
    parse_evidence_card,
    parse_visual_response,
    resolve_image_path,
    sort_records_by_sample_index,
    sync_final_csv_from_jsonl,
    to_error_record,
    write_json_atomic,
    write_jsonl,
    write_run_manifest,
    write_stage_state,
)
from eviosahs.prompts import (
    FINAL_SYSTEM_PROMPT,
    REASON_SYSTEM_PROMPT,
    SINGLE_VISUAL_QUESTION_SPEC,
    VISUAL_QUESTION_SPECS,
    VISUAL_SYSTEM_PROMPT,
    build_final_prompt,
    build_reason_prompt,
    build_visual_prompt,
    format_evidence_cards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EviOSAHS two-stage OSAHS binary experiments.")
    parser.add_argument("--input-pkl", required=True, help="Path to the local cohort pickle")
    parser.add_argument("--image-dir", required=True, help="Directory containing patient images")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument(
        "--pipeline-overlap",
        choices=["off", "same_gpu", "dual_gpu"],
        default="same_gpu",
        help="Enable ordered multithread overlap across visual, reason, and final. Use dual_gpu to pin VL and reasoner to different GPUs.",
    )
    parser.add_argument(
        "--pipeline-buffer-size",
        type=int,
        default=2,
        help="Max in-flight samples between adjacent stages when pipeline overlap is enabled.",
    )
    parser.add_argument(
        "--acceleration",
        choices=["off", "ordered_batch"],
        default="ordered_batch",
        help="Inference acceleration mode. 'ordered_batch' keeps output order aligned.",
    )
    parser.add_argument("--visual-batch-size", type=int, default=4, help="Batch size for 7 visual prompts.")
    parser.add_argument("--reason-batch-size", type=int, default=8, help="Batch size for evidence-card prompts.")
    parser.add_argument("--final-batch-size", type=int, default=8, help="Batch size for final screening prompts.")
    parser.add_argument(
        "--clinical-source",
        choices=["clean", "semantic"],
        default="clean",
        help="Use rebuilt structured clinical_summary or the raw semantic_text.",
    )
    parser.add_argument(
        "--reason-clinical-mode",
        choices=["in_reason", "final_only"],
        default="in_reason",
        help="Whether clinical information is already visible during session-level reason.",
    )
    parser.add_argument(
        "--reason-style",
        choices=["react", "summary"],
        default="react",
        help="Use session-level ReAct or a single-step evidence summary.",
    )
    parser.add_argument(
        "--evidence-strength-mode",
        choices=["on", "off"],
        default="on",
        help="Whether reason/final prompts explicitly use weak/moderate/strong evidence strength.",
    )
    parser.add_argument(
        "--final-decision-mode",
        choices=["balanced", "free"],
        default="balanced",
        help="Balanced final adjudication or freer direct yes/no decision.",
    )
    parser.add_argument(
        "--visual-decomposition",
        choices=["seven_questions", "single_pass"],
        default="seven_questions",
        help="Use the paper-style 7 visual questions or a single visual pass.",
    )
    parser.add_argument("--vl-model-path", default=None, help="Optional local path to visual-stage model weights.")
    parser.add_argument(
        "--vl-runner-type",
        choices=["qwen", "llava16"],
        default="qwen",
        help="Visual-stage model family. Use llava16 for LLaVA-1.6 visual extraction in backbone transfer experiments.",
    )
    parser.add_argument(
        "--reasoner-model-path",
        default=None,
        help="Optional local path to text reasoner weights. Qwen2.5-7B and Llama-3.1-8B-Instruct are both supported through AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--vl-device",
        default=None,
        help="Optional explicit device for the VL model, for example cuda:0.",
    )
    parser.add_argument(
        "--reasoner-device",
        default=None,
        help="Optional explicit device for the shared reasoner model. You can pass one device like cuda:1 or a multi-device string like cuda:0,1 to request one shared model load with auto device mapping.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing stage outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing outputs and rerun.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log sample errors and continue instead of stopping on the first failure.",
    )
    return parser.parse_args()


def apply_acceleration_policy(args: argparse.Namespace) -> None:
    if args.acceleration == "off":
        args.visual_batch_size = 1
        args.reason_batch_size = 1
        args.final_batch_size = 1
        args.pipeline_overlap = "off"
    if args.pipeline_overlap == "dual_gpu":
        args.vl_device = args.vl_device or "cuda:0"
        args.reasoner_device = args.reasoner_device or "cuda:1"
    if args.pipeline_buffer_size <= 0:
        raise ValueError("--pipeline-buffer-size must be positive.")


def stage_device(args: argparse.Namespace, stage: str) -> str:
    if stage == "visual":
        return args.vl_device or "auto"
    return args.reasoner_device or "auto"


def stage_extra(stage: str, args: argparse.Namespace, model_path: str, paths: dict[str, str] | None = None) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "model_path": model_path,
        "acceleration": args.acceleration,
        "clinical_source": args.clinical_source,
        "reason_clinical_mode": args.reason_clinical_mode,
        "reason_style": args.reason_style,
        "evidence_strength_mode": args.evidence_strength_mode,
        "final_decision_mode": args.final_decision_mode,
        "visual_decomposition": args.visual_decomposition,
        "vl_runner_type": args.vl_runner_type,
        "pipeline_mode": (
            "dual_gpu_multithread_ordered_pipeline"
            if args.pipeline_overlap == "dual_gpu"
            else "same_gpu_multithread_ordered_pipeline"
            if args.pipeline_overlap == "same_gpu"
            else "streaming_by_sample"
        ),
        "pipeline_overlap": args.pipeline_overlap,
    }
    if args.pipeline_overlap in {"same_gpu", "dual_gpu"}:
        extra["pipeline_threads"] = {"visual": 1, "reason": 1, "final": 1}
        extra["pipeline_buffer_size"] = args.pipeline_buffer_size
    if stage == "visual":
        extra["visual_batch_size"] = args.visual_batch_size
        extra["device"] = stage_device(args, stage)
    elif stage == "reason":
        extra["reason_batch_size"] = args.reason_batch_size
        extra["device"] = stage_device(args, stage)
    else:
        extra["final_batch_size"] = args.final_batch_size
        extra["device"] = stage_device(args, stage)
        if paths is not None:
            extra["csv_path"] = paths["final_csv"]
    return extra


def selected_visual_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.visual_decomposition == "single_pass":
        return [SINGLE_VISUAL_QUESTION_SPEC]
    return VISUAL_QUESTION_SPECS


def resolve_clinical_summary(sample_or_record: dict[str, Any], args: argparse.Namespace) -> str:
    if args.clinical_source == "semantic":
        return sample_or_record["semantic_text"]
    if "text" in sample_or_record:
        rebuilt = build_clinical_summary_from_structured_text(sample_or_record.get("text"))
    else:
        rebuilt = sample_or_record.get("clinical_summary", "")
    return rebuilt or sample_or_record["semantic_text"]


def generate_visual_batch(
    image_path: Path,
    prompts: list[str],
    vl_runner: QwenVLRunner | Llava16VisualRunner,
    batch_size: int,
) -> list[str]:
    responses: list[str] = []
    for prompt_batch in chunked(prompts, batch_size):
        try:
            responses.extend(vl_runner.generate_batch([image_path] * len(prompt_batch), prompt_batch, system_prompt=VISUAL_SYSTEM_PROMPT))
        except Exception:
            if len(prompt_batch) == 1:
                raise
            for prompt in prompt_batch:
                responses.append(vl_runner.generate(image_path=image_path, prompt=prompt, system_prompt=VISUAL_SYSTEM_PROMPT))
    return responses


def generate_reasoner_batch(
    prompts: list[str],
    reasoner: QwenReasonerRunner,
    batch_size: int,
    *,
    system_prompt: str,
) -> list[str]:
    responses: list[str] = []
    for prompt_batch in chunked(prompts, batch_size):
        try:
            responses.extend(reasoner.generate_batch(prompt_batch, system_prompt=system_prompt))
        except Exception:
            if len(prompt_batch) == 1:
                raise
            for prompt in prompt_batch:
                responses.append(reasoner.generate(prompt, system_prompt=system_prompt))
    return responses


def build_visual_record(
    sample: dict[str, Any],
    sample_index: int,
    image_dir: str | Path,
    vl_runner: QwenVLRunner | Llava16VisualRunner,
    visual_batch_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_path = resolve_image_path(image_dir, f"{sample['id']}.jpg")
    visual_specs = selected_visual_specs(args)
    prompts = [build_visual_prompt(spec["question"], spec["anatomy_target"]) for spec in visual_specs]
    raw_responses = generate_visual_batch(image_path, prompts, vl_runner, visual_batch_size)
    ensure_batch_alignment(raw_responses, len(prompts), f"visual stage {image_path.name}")

    session: list[dict[str, Any]] = []
    for session_index, (spec, raw_response) in enumerate(zip(visual_specs, raw_responses)):
        parsed = parse_visual_response(raw_response, spec["anatomy_target"])
        session.append(
            {
                "session_index": session_index,
                "visual_prompt": spec["question"],
                **parsed,
            }
        )

    return {
        "sample_index": sample_index,
        "image_path": image_path.name,
        "semantic_text": sample["semantic_text"],
        "clinical_summary": resolve_clinical_summary(sample, args),
        "numeric_label": int(sample["label"]),
        "binary_label": binary_label_from_numeric(sample["label"]),
        "session": session,
        "saved_at": now_iso(),
    }


def build_reason_record(
    record: dict[str, Any],
    reasoner: QwenReasonerRunner,
    reason_batch_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ordered_session = sorted(record["session"], key=lambda item: int(item.get("session_index", 0)))
    prompts = [
        build_reason_prompt(
            item["anatomy_target"],
            item["visual_observation"],
            item["visibility"],
            record["clinical_summary"],
            include_clinical_summary=args.reason_clinical_mode == "in_reason",
            reason_style=args.reason_style,
            include_evidence_strength=args.evidence_strength_mode == "on",
        )
        for item in ordered_session
    ]
    raw_responses = generate_reasoner_batch(
        prompts,
        reasoner,
        reason_batch_size,
        system_prompt=REASON_SYSTEM_PROMPT,
    )
    ensure_batch_alignment(raw_responses, len(prompts), f"reason stage {record['image_path']}")

    session: list[dict[str, Any]] = []
    for item, prompt, raw_response in zip(ordered_session, prompts, raw_responses):
        evidence_card, parse_status = parse_evidence_card(
            raw_response,
            default_observation=item["visual_observation"],
            default_visibility=item["visibility"],
        )
        session.append(
            {
                **item,
                "react_prompt": prompt,
                "react_raw_response": raw_response.strip(),
                "evidence_card": evidence_card,
                "evidence_parse_status": parse_status,
            }
        )

    return {
        **record,
        "session": session,
        "saved_at": now_iso(),
    }


def build_final_prompt_payload(record: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    reasoning_summary = format_evidence_cards(
        record["session"],
        include_evidence_strength=args.evidence_strength_mode == "on",
    )
    final_prompt = build_final_prompt(
        reasoning_summary,
        record["clinical_summary"],
        balanced=args.final_decision_mode == "balanced",
        include_evidence_strength=args.evidence_strength_mode == "on",
    )
    return reasoning_summary, final_prompt


def build_final_record_from_response(
    record: dict[str, Any],
    reasoning_summary: str,
    final_prompt: str,
    final_response: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predicted_label, parse_status = extract_binary_answer(final_response)
    match = predicted_label == record["binary_label"]
    final_record = {
        **record,
        "reasoning_summary": reasoning_summary,
        "final_prompt": final_prompt,
        "final_response": final_response.strip(),
        "predicted_label": predicted_label,
        "parse_status": parse_status,
        "match": match,
        "saved_at": now_iso(),
    }
    csv_row = {
        "image_path": record["image_path"],
        "gold_label": record["binary_label"],
        "predicted_label": predicted_label,
        "match": match,
        "parse_status": parse_status,
        "final_response": final_response.strip(),
    }
    return final_record, csv_row


def build_final_outputs_batch(
    record_batch: list[dict[str, Any]],
    reasoner: QwenReasonerRunner,
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    prompt_payloads = [build_final_prompt_payload(record, args) for record in record_batch]
    prompts = [prompt for _, prompt in prompt_payloads]
    responses = reasoner.generate_batch(prompts, system_prompt=FINAL_SYSTEM_PROMPT)
    ensure_batch_alignment(responses, len(prompts), "final stage")
    return [
        build_final_record_from_response(record, reasoning_summary, final_prompt, final_response)
        for record, (reasoning_summary, final_prompt), final_response in zip(record_batch, prompt_payloads, responses)
    ]


def persist_final_record(record: dict[str, Any], csv_row: dict[str, Any], paths: dict[str, str]) -> None:
    append_jsonl(record, paths["final_jsonl"])
    append_csv_row(csv_row, paths["final_csv"], FINAL_CSV_FIELDS)


def finalize_outputs(final_records: list[dict[str, Any]], paths: dict[str, str]) -> None:
    metrics = compute_binary_metrics(final_records)
    write_json_atomic(
        {
            "pred_file": paths["final_jsonl"],
            **metrics,
        },
        paths["metrics_json"],
    )
    write_json_atomic(build_analysis_payload(final_records), paths["analysis_json"])


def run_streaming_pipeline(
    dataset: list[dict[str, Any]],
    image_dir: str | Path,
    vl_runner: QwenVLRunner | Llava16VisualRunner,
    reasoner: QwenReasonerRunner,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    target_paths = {f"{sample['id']}.jpg" for sample in dataset}
    existing_visual = canonicalize_stage_records(load_existing_stage_records(paths["visual_jsonl"], target_paths, args.resume))
    existing_reason = canonicalize_stage_records(load_existing_stage_records(paths["reason_jsonl"], target_paths, args.resume))
    existing_final = canonicalize_stage_records(load_existing_stage_records(paths["final_jsonl"], target_paths, args.resume))
    if existing_visual:
        write_jsonl(existing_visual, paths["visual_jsonl"])
    if existing_reason:
        write_jsonl(existing_reason, paths["reason_jsonl"])
    if existing_final:
        write_jsonl(existing_final, paths["final_jsonl"])
        sync_final_csv_from_jsonl(existing_final, paths["final_csv"])

    visual_lookup = build_record_lookup(existing_visual)
    reason_lookup = build_record_lookup(existing_reason)
    final_lookup = build_record_lookup(existing_final)

    visual_tracker = ProgressTracker("visual", total=len(dataset), completed=len(existing_visual))
    reason_tracker = ProgressTracker("reason", total=len(dataset), completed=len(existing_reason))
    final_tracker = ProgressTracker("final", total=len(dataset), completed=len(existing_final))
    write_stage_state(paths["visual_state"], visual_tracker, status="running", output_path=paths["visual_jsonl"], error_path=paths["visual_errors"], extra=stage_extra("visual", args, vl_runner.model_name))
    write_stage_state(paths["reason_state"], reason_tracker, status="running", output_path=paths["reason_jsonl"], error_path=paths["reason_errors"], extra=stage_extra("reason", args, reasoner.model_name))
    write_stage_state(paths["final_state"], final_tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], extra=stage_extra("final", args, reasoner.model_name, paths=paths))

    for sample_index, sample in enumerate(dataset):
        image_name = f"{sample['id']}.jpg"
        if image_name in final_lookup:
            continue

        visual_record = visual_lookup.get(image_name)
        if visual_record is None:
            try:
                visual_record = build_visual_record(sample, sample_index, image_dir, vl_runner, args.visual_batch_size, args)
                append_jsonl(visual_record, paths["visual_jsonl"])
                visual_lookup[image_name] = visual_record
                visual_tracker.advance(image_name)
                write_stage_state(paths["visual_state"], visual_tracker, status="running", output_path=paths["visual_jsonl"], error_path=paths["visual_errors"], current_item=image_name, extra=stage_extra("visual", args, vl_runner.model_name))
            except Exception as exc:
                error_record = to_error_record("visual", image_name, exc)
                append_jsonl(error_record, paths["visual_errors"])
                visual_tracker.mark_error(image_name)
                write_stage_state(paths["visual_state"], visual_tracker, status="running", output_path=paths["visual_jsonl"], error_path=paths["visual_errors"], current_item=image_name, extra={**stage_extra("visual", args, vl_runner.model_name), "last_error": error_record})
                if not args.continue_on_error:
                    visual_tracker.finish()
                    reason_tracker.finish()
                    final_tracker.finish()
                    raise
                continue

        reason_record = reason_lookup.get(image_name)
        if reason_record is None:
            try:
                reason_record = build_reason_record(visual_record, reasoner, args.reason_batch_size, args)
                append_jsonl(reason_record, paths["reason_jsonl"])
                reason_lookup[image_name] = reason_record
                reason_tracker.advance(image_name)
                write_stage_state(paths["reason_state"], reason_tracker, status="running", output_path=paths["reason_jsonl"], error_path=paths["reason_errors"], current_item=image_name, extra=stage_extra("reason", args, reasoner.model_name))
            except Exception as exc:
                error_record = to_error_record("reason", image_name, exc)
                append_jsonl(error_record, paths["reason_errors"])
                reason_tracker.mark_error(image_name)
                write_stage_state(paths["reason_state"], reason_tracker, status="running", output_path=paths["reason_jsonl"], error_path=paths["reason_errors"], current_item=image_name, extra={**stage_extra("reason", args, reasoner.model_name), "last_error": error_record})
                if not args.continue_on_error:
                    visual_tracker.finish()
                    reason_tracker.finish()
                    final_tracker.finish()
                    raise
                continue

        try:
            final_record, csv_row = build_final_outputs_batch([reason_record], reasoner, args)[0]
            persist_final_record(final_record, csv_row, paths)
            final_lookup[image_name] = final_record
            final_tracker.advance(image_name)
            write_stage_state(paths["final_state"], final_tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], current_item=image_name, extra=stage_extra("final", args, reasoner.model_name, paths=paths))
        except Exception as exc:
            error_record = to_error_record("final", image_name, exc)
            append_jsonl(error_record, paths["final_errors"])
            final_tracker.mark_error(image_name)
            write_stage_state(paths["final_state"], final_tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], current_item=image_name, extra={**stage_extra("final", args, reasoner.model_name, paths=paths), "last_error": error_record})
            if not args.continue_on_error:
                visual_tracker.finish()
                reason_tracker.finish()
                final_tracker.finish()
                raise

    visual_tracker.finish()
    reason_tracker.finish()
    final_tracker.finish()
    final_visual_records = canonicalize_stage_records(list(visual_lookup.values()))
    final_reason_records = canonicalize_stage_records(list(reason_lookup.values()))
    final_final_records = canonicalize_stage_records(list(final_lookup.values()))
    write_jsonl(final_visual_records, paths["visual_jsonl"])
    write_jsonl(final_reason_records, paths["reason_jsonl"])
    write_jsonl(final_final_records, paths["final_jsonl"])
    sync_final_csv_from_jsonl(final_final_records, paths["final_csv"])
    write_stage_state(paths["visual_state"], visual_tracker, status="completed_with_errors" if visual_tracker.errors else "completed", output_path=paths["visual_jsonl"], error_path=paths["visual_errors"], extra=stage_extra("visual", args, vl_runner.model_name))
    write_stage_state(paths["reason_state"], reason_tracker, status="completed_with_errors" if reason_tracker.errors else "completed", output_path=paths["reason_jsonl"], error_path=paths["reason_errors"], extra=stage_extra("reason", args, reasoner.model_name))
    write_stage_state(paths["final_state"], final_tracker, status="completed_with_errors" if final_tracker.errors else "completed", output_path=paths["final_jsonl"], error_path=paths["final_errors"], extra=stage_extra("final", args, reasoner.model_name, paths=paths))
    return final_final_records


def run_ordered_pipeline(
    dataset: list[dict[str, Any]],
    image_dir: str | Path,
    vl_runner: QwenVLRunner | Llava16VisualRunner,
    reasoner: QwenReasonerRunner,
    paths: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    target_paths = {f"{sample['id']}.jpg" for sample in dataset}
    existing_visual = canonicalize_stage_records(load_existing_stage_records(paths["visual_jsonl"], target_paths, args.resume))
    existing_reason = canonicalize_stage_records(load_existing_stage_records(paths["reason_jsonl"], target_paths, args.resume))
    existing_final = canonicalize_stage_records(load_existing_stage_records(paths["final_jsonl"], target_paths, args.resume))
    if existing_visual:
        write_jsonl(existing_visual, paths["visual_jsonl"])
    if existing_reason:
        write_jsonl(existing_reason, paths["reason_jsonl"])
    if existing_final:
        write_jsonl(existing_final, paths["final_jsonl"])
        sync_final_csv_from_jsonl(existing_final, paths["final_csv"])

    visual_lookup = build_record_lookup(existing_visual)
    reason_lookup = build_record_lookup(existing_reason)
    final_lookup = build_record_lookup(existing_final)

    lookup_lock = threading.Lock()
    status_lock = threading.Lock()
    visual_tracker = ProgressTracker("visual", total=len(dataset), completed=len(existing_visual))
    reason_tracker = ProgressTracker("reason", total=len(dataset), completed=len(existing_reason))
    final_tracker = ProgressTracker("final", total=len(dataset), completed=len(existing_final))

    with status_lock:
        write_stage_state(paths["visual_state"], visual_tracker, status="running", output_path=paths["visual_jsonl"], error_path=paths["visual_errors"], extra=stage_extra("visual", args, vl_runner.model_name))
        write_stage_state(paths["reason_state"], reason_tracker, status="running", output_path=paths["reason_jsonl"], error_path=paths["reason_errors"], extra=stage_extra("reason", args, reasoner.model_name))
        write_stage_state(paths["final_state"], final_tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], extra=stage_extra("final", args, reasoner.model_name, paths=paths))

    vl_runner.ensure_loaded()
    reasoner.ensure_loaded()

    reason_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=args.pipeline_buffer_size)
    final_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=args.pipeline_buffer_size)
    stop_event = threading.Event()
    visual_done = threading.Event()
    reason_done = threading.Event()
    fatal_exceptions: list[Exception] = []
    fatal_lock = threading.Lock()

    def record_fatal(exc: Exception) -> None:
        with fatal_lock:
            if not fatal_exceptions:
                fatal_exceptions.append(exc)
        stop_event.set()

    def put_queue_item(target_queue: queue.Queue[dict[str, Any]], item: dict[str, Any]) -> bool:
        while True:
            if stop_event.is_set():
                return False
            try:
                target_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue

    def record_stage_error(stage: str, image_name: str, exc: Exception) -> None:
        error_record = to_error_record(stage, image_name, exc)
        error_path = {
            "visual": paths["visual_errors"],
            "reason": paths["reason_errors"],
            "final": paths["final_errors"],
        }[stage]
        tracker = {
            "visual": visual_tracker,
            "reason": reason_tracker,
            "final": final_tracker,
        }[stage]
        state_path = {
            "visual": paths["visual_state"],
            "reason": paths["reason_state"],
            "final": paths["final_state"],
        }[stage]
        output_path = {
            "visual": paths["visual_jsonl"],
            "reason": paths["reason_jsonl"],
            "final": paths["final_jsonl"],
        }[stage]
        model_name = vl_runner.model_name if stage == "visual" else reasoner.model_name
        with status_lock:
            append_jsonl(error_record, error_path)
            tracker.mark_error(image_name)
            write_stage_state(
                state_path,
                tracker,
                status="running",
                output_path=output_path,
                error_path=error_path,
                current_item=image_name,
                extra={**stage_extra(stage, args, model_name, paths=paths if stage == "final" else None), "last_error": error_record},
            )

    def visual_worker() -> None:
        try:
            for sample_index, sample in enumerate(dataset):
                if stop_event.is_set():
                    break
                image_name = f"{sample['id']}.jpg"
                with lookup_lock:
                    final_record = final_lookup.get(image_name)
                    reason_record = reason_lookup.get(image_name)
                    visual_record = visual_lookup.get(image_name)
                if final_record is not None:
                    continue
                if reason_record is not None:
                    if not put_queue_item(reason_queue, {"sample_index": sample_index, "image_path": image_name, "reason_record": reason_record}):
                        break
                    continue
                if visual_record is None:
                    try:
                        visual_record = build_visual_record(sample, sample_index, image_dir, vl_runner, args.visual_batch_size, args)
                        with status_lock:
                            append_jsonl(visual_record, paths["visual_jsonl"])
                            with lookup_lock:
                                visual_lookup[image_name] = visual_record
                            visual_tracker.advance(image_name)
                            write_stage_state(paths["visual_state"], visual_tracker, status="running", output_path=paths["visual_jsonl"], error_path=paths["visual_errors"], current_item=image_name, extra=stage_extra("visual", args, vl_runner.model_name))
                    except Exception as exc:
                        record_stage_error("visual", image_name, exc)
                        if not args.continue_on_error:
                            record_fatal(exc)
                            break
                        continue
                if not put_queue_item(reason_queue, {"sample_index": sample_index, "image_path": image_name, "visual_record": visual_record}):
                    break
        except Exception as exc:
            record_fatal(exc)
        finally:
            visual_done.set()

    def reason_worker() -> None:
        try:
            while True:
                if stop_event.is_set() and visual_done.is_set() and reason_queue.empty():
                    break
                try:
                    item = reason_queue.get(timeout=0.1)
                except queue.Empty:
                    if visual_done.is_set():
                        break
                    continue
                image_name = str(item["image_path"])
                try:
                    reason_record = item.get("reason_record")
                    if reason_record is None:
                        try:
                            reason_record = build_reason_record(item["visual_record"], reasoner, args.reason_batch_size, args)
                            with status_lock:
                                append_jsonl(reason_record, paths["reason_jsonl"])
                                with lookup_lock:
                                    reason_lookup[image_name] = reason_record
                                reason_tracker.advance(image_name)
                                write_stage_state(paths["reason_state"], reason_tracker, status="running", output_path=paths["reason_jsonl"], error_path=paths["reason_errors"], current_item=image_name, extra=stage_extra("reason", args, reasoner.model_name))
                        except Exception as exc:
                            record_stage_error("reason", image_name, exc)
                            if not args.continue_on_error:
                                record_fatal(exc)
                            continue
                    if not put_queue_item(final_queue, {"sample_index": item["sample_index"], "image_path": image_name, "reason_record": reason_record}):
                        break
                finally:
                    reason_queue.task_done()
        except Exception as exc:
            record_fatal(exc)
        finally:
            reason_done.set()

    def persist_final_success(final_record: dict[str, Any], csv_row: dict[str, Any]) -> None:
        with status_lock:
            persist_final_record(final_record, csv_row, paths)
            with lookup_lock:
                final_lookup[final_record["image_path"]] = final_record
            final_tracker.advance(final_record["image_path"])
            write_stage_state(paths["final_state"], final_tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], current_item=final_record["image_path"], extra=stage_extra("final", args, reasoner.model_name, paths=paths))

    def final_worker() -> None:
        try:
            while True:
                if stop_event.is_set() and reason_done.is_set() and final_queue.empty():
                    break
                try:
                    first_item = final_queue.get(timeout=0.1)
                except queue.Empty:
                    if reason_done.is_set():
                        break
                    continue

                batch_items = [first_item]
                while len(batch_items) < args.final_batch_size:
                    try:
                        batch_items.append(final_queue.get_nowait())
                    except queue.Empty:
                        break

                try:
                    record_batch = [item["reason_record"] for item in batch_items]
                    try:
                        batch_outputs = build_final_outputs_batch(record_batch, reasoner, args)
                        for final_record, csv_row in batch_outputs:
                            persist_final_success(final_record, csv_row)
                    except Exception:
                        for queued_item in batch_items:
                            image_name = str(queued_item["image_path"])
                            try:
                                final_record, csv_row = build_final_outputs_batch([queued_item["reason_record"]], reasoner, args)[0]
                                persist_final_success(final_record, csv_row)
                            except Exception as exc:
                                record_stage_error("final", image_name, exc)
                                if not args.continue_on_error:
                                    record_fatal(exc)
                                    return
                finally:
                    for _ in batch_items:
                        final_queue.task_done()
        except Exception as exc:
            record_fatal(exc)

    visual_thread = threading.Thread(target=visual_worker, name="eviosahs-visual")
    reason_thread = threading.Thread(target=reason_worker, name="eviosahs-reason")
    final_thread = threading.Thread(target=final_worker, name="eviosahs-final")
    visual_thread.start()
    reason_thread.start()
    final_thread.start()
    visual_thread.join()
    vl_runner.close()
    reason_thread.join()
    final_thread.join()

    if fatal_exceptions:
        visual_tracker.finish()
        reason_tracker.finish()
        final_tracker.finish()
        raise fatal_exceptions[0]

    visual_tracker.finish()
    reason_tracker.finish()
    final_tracker.finish()
    final_visual_records = canonicalize_stage_records(list(visual_lookup.values()))
    final_reason_records = canonicalize_stage_records(list(reason_lookup.values()))
    final_final_records = canonicalize_stage_records(list(final_lookup.values()))
    write_jsonl(final_visual_records, paths["visual_jsonl"])
    write_jsonl(final_reason_records, paths["reason_jsonl"])
    write_jsonl(final_final_records, paths["final_jsonl"])
    sync_final_csv_from_jsonl(final_final_records, paths["final_csv"])
    with status_lock:
        write_stage_state(paths["visual_state"], visual_tracker, status="completed_with_errors" if visual_tracker.errors else "completed", output_path=paths["visual_jsonl"], error_path=paths["visual_errors"], extra=stage_extra("visual", args, vl_runner.model_name))
        write_stage_state(paths["reason_state"], reason_tracker, status="completed_with_errors" if reason_tracker.errors else "completed", output_path=paths["reason_jsonl"], error_path=paths["reason_errors"], extra=stage_extra("reason", args, reasoner.model_name))
        write_stage_state(paths["final_state"], final_tracker, status="completed_with_errors" if final_tracker.errors else "completed", output_path=paths["final_jsonl"], error_path=paths["final_errors"], extra=stage_extra("final", args, reasoner.model_name, paths=paths))
    return final_final_records


def main() -> None:
    args = parse_args()
    apply_acceleration_policy(args)
    paths = build_output_paths(args.output_dir)
    enforce_output_policy(paths, args.resume, args.overwrite)
    dataset = limit_records(load_pickle_dataset(args.input_pkl), args.limit)
    if args.vl_runner_type == "llava16":
        vl_runner = Llava16VisualRunner(model_name=args.vl_model_path, device_map=build_device_map(args.vl_device))
    else:
        vl_runner = QwenVLRunner(model_name=args.vl_model_path, device_map=build_device_map(args.vl_device))
    reasoner = QwenReasonerRunner(
        model_name=args.reasoner_model_path,
        device_map=build_device_map(args.reasoner_device),
    )
    write_run_manifest(
        paths["run_manifest"],
        script_name="two_stage_binary.py",
        args=vars(args),
        resolved_models={
            "vl_model": vl_runner.model_name,
            "vl_runner_type": args.vl_runner_type,
            "reasoner_model": reasoner.model_name,
            "vl_device": args.vl_device or "auto",
            "reasoner_device": args.reasoner_device or "auto",
        },
        output_dir=args.output_dir,
    )
    try:
        if args.pipeline_overlap in {"same_gpu", "dual_gpu"}:
            final_records = run_ordered_pipeline(dataset, args.image_dir, vl_runner, reasoner, paths, args)
        else:
            final_records = run_streaming_pipeline(dataset, args.image_dir, vl_runner, reasoner, paths, args)
        finalize_outputs(final_records, paths)
    finally:
        vl_runner.close()
        reasoner.close()


if __name__ == "__main__":
    main()
