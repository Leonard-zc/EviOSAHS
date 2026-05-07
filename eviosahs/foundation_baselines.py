from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from eviosahs.analyze_run import build_analysis_payload
from eviosahs.models import (
    FINAL_CSV_FIELDS,
    ProgressTracker,
    QwenReasonerRunner,
    QwenVLRunner,
    append_csv_row,
    append_jsonl,
    binary_label_from_numeric,
    build_clinical_summary_from_structured_text,
    build_device_map,
    compute_binary_metrics,
    enforce_output_policy,
    extract_binary_answer,
    now_iso,
    build_output_paths,
    load_existing_stage_records,
    load_pickle_dataset,
    parse_device_list,
    configure_generation_config,
    resolve_image_path,
    sync_final_csv_from_jsonl,
    to_error_record,
    write_json_atomic,
    write_run_manifest,
    write_stage_state,
)
from eviosahs.prompts import (
    FINAL_SYSTEM_PROMPT,
    build_clinical_only_prompt,
    build_direct_multimodal_prompt,
)


DEFAULT_LLaVA_16_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
DEFAULT_INSTRUCTBLIP_MODEL = "Salesforce/instructblip-vicuna-7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run foundation-model direct baselines for OSAHS screening.")
    parser.add_argument("--input-pkl", required=True, help="Path to the local cohort pickle")
    parser.add_argument("--image-dir", required=True, help="Directory containing patient images")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    parser.add_argument(
        "--method",
        required=True,
        choices=["clinical_only_qwen_text", "qwen_direct", "llava16_direct", "instructblip_direct"],
        help="Foundation baseline to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument(
        "--clinical-source",
        choices=["clean", "semantic"],
        default="clean",
        help="Use rebuilt structured clinical_summary or the raw semantic_text.",
    )
    parser.add_argument("--qwen-vl-model-path", default=None, help="Optional local path to Qwen2.5-VL weights")
    parser.add_argument("--qwen-text-model-path", default=None, help="Optional local path to Qwen2.5 text weights")
    parser.add_argument("--llava-model-path", default=DEFAULT_LLaVA_16_MODEL, help="HF id or local path for LLaVA-1.6")
    parser.add_argument("--instructblip-model-path", default=DEFAULT_INSTRUCTBLIP_MODEL, help="HF id or local path for InstructBLIP")
    parser.add_argument("--device", default=None, help="Target device, for example cuda:0")
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing outputs and rerun.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after per-sample errors.")
    return parser.parse_args()


def resolve_clinical_summary(sample: dict[str, Any], clinical_source: str) -> str:
    if clinical_source == "semantic":
        return sample["semantic_text"]
    rebuilt = build_clinical_summary_from_structured_text(sample.get("text"))
    return rebuilt or sample["semantic_text"]


def resolve_device(device: str | None) -> str:
    devices = parse_device_list(device)
    if devices:
        return devices[0]
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class InstructBlipDirectRunner:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        import torch
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

        self._processor = InstructBlipProcessor.from_pretrained(self.model_name)
        self._model = InstructBlipForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self._model.to(self.device)
        self._model.eval()
        configure_generation_config(self._model, do_sample=False)

    def generate(self, image_path: Path, prompt: str) -> str:
        self._lazy_load()
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


class Llava16DirectRunner:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self._processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self._model.to(self.device)
        self._model.eval()
        configure_generation_config(self._model, do_sample=False)

    def generate(self, image_path: Path, prompt: str) -> str:
        self._lazy_load()
        import torch

        image = Image.open(image_path).convert("RGB")
        llava_prompt = f"[INST] <image>\n{prompt} [/INST]"
        inputs = self._processor(text=llava_prompt, images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def build_record(sample: dict[str, Any], sample_index: int, image_dir: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    image_name = f"{sample['id']}.jpg"
    return {
        "sample_index": sample_index,
        "image_path": image_name,
        "resolved_image_path": str(resolve_image_path(image_dir, image_name)),
        "semantic_text": sample["semantic_text"],
        "clinical_summary": resolve_clinical_summary(sample, args.clinical_source),
        "numeric_label": int(sample["label"]),
        "binary_label": binary_label_from_numeric(sample["label"]),
    }


def persist_final_record(record: dict[str, Any], paths: dict[str, str]) -> None:
    append_jsonl(record, paths["final_jsonl"])
    append_csv_row(
        {
            "image_path": record["image_path"],
            "gold_label": record["binary_label"],
            "predicted_label": record["predicted_label"],
            "match": record["match"],
            "parse_status": record["parse_status"],
            "final_response": record["final_response"],
        },
        paths["final_csv"],
        FINAL_CSV_FIELDS,
    )


def finalize_outputs(final_records: list[dict[str, Any]], paths: dict[str, str]) -> None:
    metrics = compute_binary_metrics(final_records)
    write_json_atomic({"pred_file": paths["final_jsonl"], **metrics}, paths["metrics_json"])
    write_json_atomic(build_analysis_payload(final_records), paths["analysis_json"])


def build_final_record(record: dict[str, Any], final_prompt: str, final_response: str, method: str) -> dict[str, Any]:
    predicted_label, parse_status = extract_binary_answer(final_response)
    return {
        **record,
        "method": method,
        "final_prompt": final_prompt,
        "final_response": final_response.strip(),
        "predicted_label": predicted_label,
        "parse_status": parse_status,
        "match": predicted_label == record["binary_label"],
        "saved_at": now_iso(),
    }


def run_clinical_only_qwen(dataset: list[dict[str, Any]], args: argparse.Namespace, paths: dict[str, str]) -> list[dict[str, Any]]:
    runner = QwenReasonerRunner(model_name=args.qwen_text_model_path, device_map=build_device_map(args.device))
    target_paths = {f"{sample['id']}.jpg" for sample in dataset}
    existing = load_existing_stage_records(paths["final_jsonl"], target_paths, args.resume)
    final_lookup = {record["image_path"]: record for record in existing}
    if existing:
        sync_final_csv_from_jsonl(existing, paths["final_csv"])
    tracker = ProgressTracker("final", total=len(dataset), completed=len(existing))
    write_stage_state(paths["final_state"], tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], extra={"method": args.method, "model_path": runner.model_name})

    for sample_index, sample in enumerate(dataset):
        image_name = f"{sample['id']}.jpg"
        if image_name in final_lookup:
            continue
        base_record = build_record(sample, sample_index, args.image_dir, args)
        prompt = build_clinical_only_prompt(base_record["clinical_summary"], balanced=True)
        try:
            response = runner.generate(prompt, system_prompt=FINAL_SYSTEM_PROMPT)
            final_record = build_final_record(base_record, prompt, response, args.method)
            persist_final_record(final_record, paths)
            final_lookup[image_name] = final_record
            tracker.advance(image_name)
            write_stage_state(paths["final_state"], tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], current_item=image_name, extra={"method": args.method, "model_path": runner.model_name})
        except Exception as exc:
            append_jsonl(to_error_record("final", image_name, exc), paths["final_errors"])
            tracker.mark_error(image_name)
            if not args.continue_on_error:
                raise
    final_records = sorted(final_lookup.values(), key=lambda record: int(record.get("sample_index", 10**9)))
    finalize_outputs(final_records, paths)
    sync_final_csv_from_jsonl(final_records, paths["final_csv"])
    write_stage_state(paths["final_state"], tracker, status="completed_with_errors" if tracker.errors else "completed", output_path=paths["final_jsonl"], error_path=paths["final_errors"], extra={"method": args.method, "model_path": runner.model_name})
    return final_records


def run_multimodal_direct(dataset: list[dict[str, Any]], args: argparse.Namespace, paths: dict[str, str]) -> list[dict[str, Any]]:
    if args.method == "qwen_direct":
        runner = QwenVLRunner(model_name=args.qwen_vl_model_path, device_map=build_device_map(args.device))
        model_path = runner.model_name
        generate = lambda image_path, prompt: runner.generate(image_path=image_path, prompt=prompt, system_prompt=FINAL_SYSTEM_PROMPT)
    elif args.method == "instructblip_direct":
        runner = InstructBlipDirectRunner(args.instructblip_model_path, device=args.device)
        model_path = args.instructblip_model_path
        generate = lambda image_path, prompt: runner.generate(image_path, prompt)
    elif args.method == "llava16_direct":
        runner = Llava16DirectRunner(args.llava_model_path, device=args.device)
        model_path = args.llava_model_path
        generate = lambda image_path, prompt: runner.generate(image_path, prompt)
    else:
        raise ValueError(f"Unsupported direct method: {args.method}")

    target_paths = {f"{sample['id']}.jpg" for sample in dataset}
    existing = load_existing_stage_records(paths["final_jsonl"], target_paths, args.resume)
    final_lookup = {record["image_path"]: record for record in existing}
    if existing:
        sync_final_csv_from_jsonl(existing, paths["final_csv"])
    tracker = ProgressTracker("final", total=len(dataset), completed=len(existing))
    write_stage_state(paths["final_state"], tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], extra={"method": args.method, "model_path": model_path})

    for sample_index, sample in enumerate(dataset):
        image_name = f"{sample['id']}.jpg"
        if image_name in final_lookup:
            continue
        base_record = build_record(sample, sample_index, args.image_dir, args)
        prompt = build_direct_multimodal_prompt(base_record["clinical_summary"], balanced=True)
        try:
            response = generate(Path(base_record["resolved_image_path"]), prompt)
            final_record = build_final_record(base_record, prompt, response, args.method)
            persist_final_record(final_record, paths)
            final_lookup[image_name] = final_record
            tracker.advance(image_name)
            write_stage_state(paths["final_state"], tracker, status="running", output_path=paths["final_jsonl"], error_path=paths["final_errors"], current_item=image_name, extra={"method": args.method, "model_path": model_path})
        except Exception as exc:
            append_jsonl(to_error_record("final", image_name, exc), paths["final_errors"])
            tracker.mark_error(image_name)
            if not args.continue_on_error:
                raise

    final_records = sorted(final_lookup.values(), key=lambda record: int(record.get("sample_index", 10**9)))
    finalize_outputs(final_records, paths)
    sync_final_csv_from_jsonl(final_records, paths["final_csv"])
    write_stage_state(paths["final_state"], tracker, status="completed_with_errors" if tracker.errors else "completed", output_path=paths["final_jsonl"], error_path=paths["final_errors"], extra={"method": args.method, "model_path": model_path})
    return final_records


def main() -> None:
    args = parse_args()
    paths = build_output_paths(args.output_dir)
    enforce_output_policy(paths, args.resume, args.overwrite)
    dataset = load_pickle_dataset(args.input_pkl)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    resolved_models = {
        "qwen_vl": args.qwen_vl_model_path or "",
        "qwen_text": args.qwen_text_model_path or "",
        "llava16": args.llava_model_path,
        "instructblip": args.instructblip_model_path,
    }
    write_run_manifest(
        paths["run_manifest"],
        script_name=Path(__file__).name,
        args=vars(args),
        resolved_models=resolved_models,
        output_dir=args.output_dir,
    )

    if args.method == "clinical_only_qwen_text":
        run_clinical_only_qwen(dataset, args, paths)
    else:
        run_multimodal_direct(dataset, args, paths)


if __name__ == "__main__":
    main()
