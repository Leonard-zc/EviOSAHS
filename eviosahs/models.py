from __future__ import annotations

import csv
import gc
import json
import os
import pickle
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_QWEN_MODEL_ROOT = Path(
    "/cpfs01/projects-HDD/cfff-7361474ef8eb_HDD/huangjingjing/model/Qwen"
)
DEFAULT_VL_HF_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_REASONER_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LLAVA16_HF_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

FINAL_CSV_FIELDS = [
    "image_path",
    "gold_label",
    "predicted_label",
    "match",
    "parse_status",
    "final_response",
]

STRUCTURED_FEATURE_NAMES = [
    "sex_male",
    "age",
    "neck_circumference_cm",
    "bmi",
    "waist_hip_ratio",
    "hypertension",
    "diabetes",
    "heart_disease",
    "hyperlipidemia",
]

VISUAL_NOISE_KEYWORDS = [
    "shirt",
    "striped shirt",
    "glasses",
    "background",
    "pet",
    "dog",
    "medical device",
    "upside-down",
    "abstract pattern",
    "camera angle",
]


@dataclass
class GenerationSettings:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05


def _iter_candidate_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []

    candidates: list[Path] = []
    for child in root.iterdir():
        if child.is_dir():
            candidates.append(child)
            for nested in child.iterdir():
                if nested.is_dir():
                    candidates.append(nested)
    return candidates


def _find_local_model_dir(root: Path, name_hints: list[str]) -> Path | None:
    lowered_hints = [hint.lower() for hint in name_hints]

    for hint in name_hints:
        direct = root / hint
        if direct.exists():
            return direct

    candidates = list(_iter_candidate_dirs(root))
    for hint in lowered_hints:
        for candidate in candidates:
            if hint in candidate.name.lower():
                return candidate

    return None


def resolve_model_source(
    explicit_path: str | None,
    env_var_name: str,
    default_hf_model: str,
    name_hints: list[str],
) -> str:
    if explicit_path:
        return explicit_path

    env_path = os.environ.get(env_var_name)
    if env_path:
        return env_path

    root_override = os.environ.get("QWEN_MODEL_ROOT")
    model_root = Path(root_override) if root_override else DEFAULT_QWEN_MODEL_ROOT
    local_model_dir = _find_local_model_dir(model_root, name_hints)
    if local_model_dir is not None:
        return str(local_model_dir)

    return default_hf_model


def parse_device_list(device_spec: str | None) -> list[str]:
    if not device_spec:
        return []

    parts = [part.strip() for part in str(device_spec).split(",") if part.strip()]
    if not parts:
        return []

    devices: list[str] = []
    for part in parts:
        if part.isdigit():
            devices.append(f"cuda:{part}")
        else:
            devices.append(part)
    return devices


def build_device_map(device: str | None, default: str = "auto") -> Any:
    if not device:
        return default
    devices = parse_device_list(device)
    if not devices:
        return default
    if len(devices) > 1:
        return default
    return {"": devices[0]}


def first_device_from_map(device_map: Any) -> str | None:
    if isinstance(device_map, dict):
        for device in device_map.values():
            if isinstance(device, str):
                return device
    if isinstance(device_map, str) and device_map != "auto":
        return device_map
    return None


def load_pickle_dataset(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def binary_label_from_numeric(label: int) -> str:
    return "no" if int(label) == 0 else "yes"


def _coerce_float(value: Any) -> float:
    return float(str(value).strip())


def _coerce_int_flag(value: Any) -> int:
    return int(float(str(value).strip()))


def normalize_gender(raw_gender: Any) -> str:
    text = str(raw_gender).strip().lower()
    if text in {"male", "m"}:
        return "male"
    if text in {"female", "f"}:
        return "female"
    return text or "unknown"


def bmi_category(bmi: float) -> str:
    if bmi >= 40:
        return "morbid obesity"
    if bmi >= 30:
        return "obesity"
    if bmi >= 25:
        return "overweight"
    if bmi >= 18.5:
        return "healthy weight"
    return "underweight"


def whr_category(whr: float) -> str:
    if whr >= 1.0:
        return "markedly elevated"
    if whr >= 0.95:
        return "elevated"
    if whr >= 0.9:
        return "borderline high"
    return "not elevated"


def format_comorbidity_summary(flags: dict[str, int]) -> str:
    positive = [label for label, flag in flags.items() if flag == 1]
    negative = [label for label, flag in flags.items() if flag == 0]
    if positive and negative:
        return (
            f"medical history of {', '.join(positive)}, "
            f"but not {', '.join(negative)}"
        )
    if positive:
        return f"medical history of {', '.join(positive)}"
    return f"no history of {', '.join(negative)}"


def build_clinical_summary_from_structured_text(text_fields: Any) -> str:
    if not isinstance(text_fields, (list, tuple)) or len(text_fields) < 9:
        return ""

    gender = normalize_gender(text_fields[0])
    neck_circumference = _coerce_float(text_fields[1])
    bmi = _coerce_float(text_fields[2])
    hypertension = _coerce_int_flag(text_fields[3])
    diabetes = _coerce_int_flag(text_fields[4])
    heart_disease = _coerce_int_flag(text_fields[5])
    hyperlipidemia = _coerce_int_flag(text_fields[6])
    age = int(round(_coerce_float(text_fields[7])))
    whr = _coerce_float(text_fields[8])

    positive_risk: list[str] = []
    protective_context: list[str] = []

    bmi_text = bmi_category(bmi)
    if bmi >= 25:
        positive_risk.append(f"BMI category: {bmi_text}")
    else:
        protective_context.append(f"BMI category: {bmi_text}")

    if hypertension:
        positive_risk.append("history of hypertension")
    else:
        protective_context.append("no history of hypertension")
    if diabetes:
        positive_risk.append("history of diabetes")
    else:
        protective_context.append("no history of diabetes")
    if heart_disease:
        positive_risk.append("history of heart disease")
    else:
        protective_context.append("no history of heart disease")
    if hyperlipidemia:
        positive_risk.append("history of hyperlipidemia")
    else:
        protective_context.append("no history of hyperlipidemia")

    if not positive_risk:
        positive_risk.append("none explicitly recorded")

    return "\n".join(
        [
            "PatientProfile:",
            f"- Age: {age}",
            f"- Sex: {gender}",
            f"- NeckCircumferenceCm: {neck_circumference:.1f}",
            f"- BMI: {bmi:.1f}",
            f"- BMICategory: {bmi_text}",
            f"- WaistHipRatio: {whr:.2f}",
            f"- WaistHipRatioCategory: {whr_category(whr)}",
            "PositiveClinicalRisk:",
            *[f"- {item}" for item in positive_risk],
            "ProtectiveClinicalContext:",
            *[f"- {item}" for item in protective_context],
        ]
    )


def build_structured_feature_dict(text_fields: Any) -> dict[str, float]:
    if not isinstance(text_fields, (list, tuple)) or len(text_fields) < 9:
        return {name: 0.0 for name in STRUCTURED_FEATURE_NAMES}

    gender = normalize_gender(text_fields[0])
    neck_circumference = _coerce_float(text_fields[1])
    bmi = _coerce_float(text_fields[2])
    hypertension = _coerce_int_flag(text_fields[3])
    diabetes = _coerce_int_flag(text_fields[4])
    heart_disease = _coerce_int_flag(text_fields[5])
    hyperlipidemia = _coerce_int_flag(text_fields[6])
    age = int(round(_coerce_float(text_fields[7])))
    whr = _coerce_float(text_fields[8])
    return {
        "sex_male": 1.0 if gender == "male" else 0.0,
        "age": float(age),
        "neck_circumference_cm": float(neck_circumference),
        "bmi": float(bmi),
        "waist_hip_ratio": float(whr),
        "hypertension": float(hypertension),
        "diabetes": float(diabetes),
        "heart_disease": float(heart_disease),
        "hyperlipidemia": float(hyperlipidemia),
    }


def build_structured_feature_vector(text_fields: Any) -> list[float]:
    feature_dict = build_structured_feature_dict(text_fields)
    return [float(feature_dict[name]) for name in STRUCTURED_FEATURE_NAMES]


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(output_path)


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path, strict: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if strict:
                    raise
    return records


def write_csv(records: list[dict[str, Any]], path: str | Path, fieldnames: list[str]) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def append_csv_row(record: dict[str, Any], path: str | Path, fieldnames: list[str]) -> None:
    output_path = ensure_parent_dir(path)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def resolve_image_path(image_dir: str | Path, image_name: str) -> Path:
    image_path = Path(image_dir) / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path.resolve()


def limit_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[:limit]


def remove_file(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_record_index(records: list[dict[str, Any]], key: str = "image_path") -> set[str]:
    return {str(record[key]) for record in records if key in record}


def build_record_lookup(records: list[dict[str, Any]], key: str = "image_path") -> dict[str, dict[str, Any]]:
    return {str(record[key]): record for record in records if key in record}


def chunked(records: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def ensure_batch_alignment(responses: list[str], expected_count: int, context: str) -> None:
    if len(responses) != expected_count:
        raise ValueError(
            f"{context} returned {len(responses)} responses, expected {expected_count}. "
            "This would break record ordering, so the batch is rejected."
        )


def sort_records_by_sample_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(record: dict[str, Any]) -> tuple[int, str]:
        sample_index = record.get("sample_index")
        if sample_index is None:
            return (10**9, str(record.get("image_path", "")))
        return (int(sample_index), str(record.get("image_path", "")))

    return sorted(records, key=sort_key)


def canonicalize_stage_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_records = sort_records_by_sample_index(records)
    for record in canonical_records:
        if "session" in record:
            record["session"] = sorted(record["session"], key=lambda item: int(item.get("session_index", 0)))
    return canonical_records


def configure_left_padding(tokenizer: Any, model: Any | None = None) -> None:
    if tokenizer is None:
        return

    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if model is not None and pad_token_id is not None and hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = pad_token_id


def configure_generation_config(
    model: Any | None,
    *,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
) -> None:
    if model is None or not hasattr(model, "generation_config"):
        return

    generation_config = model.generation_config
    generation_config.do_sample = do_sample

    if repetition_penalty is not None and hasattr(generation_config, "repetition_penalty"):
        generation_config.repetition_penalty = repetition_penalty

    if do_sample:
        if temperature is not None and hasattr(generation_config, "temperature"):
            generation_config.temperature = temperature
        if top_p is not None and hasattr(generation_config, "top_p"):
            generation_config.top_p = top_p
        return

    # Reset common sampling-only fields to greedy defaults so transformers
    # does not emit "invalid generation flags may be ignored" warnings.
    greedy_defaults: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "typical_p": 1.0,
        "min_p": None,
    }
    for field_name, field_value in greedy_defaults.items():
        if hasattr(generation_config, field_name):
            setattr(generation_config, field_name, field_value)


class ProgressTracker:
    def __init__(self, stage: str, total: int, completed: int = 0, errors: int = 0, bar_width: int = 24) -> None:
        self.stage = stage
        self.total = total
        self.completed = completed
        self.errors = errors
        self.attempted = completed
        self.bar_width = bar_width
        self.started_at = time.time()
        self.initial_completed = completed
        self.initial_attempted = completed
        self._interactive = sys.stdout.isatty()
        self._lock = threading.Lock()

    def _remaining_seconds(self) -> float | None:
        newly_attempted = self.attempted - self.initial_attempted
        if newly_attempted <= 0:
            return None
        elapsed = time.time() - self.started_at
        rate = newly_attempted / elapsed if elapsed > 0 else 0.0
        if rate <= 0:
            return None
        return (self.total - self.attempted) / rate

    def _rate(self) -> float:
        elapsed = time.time() - self.started_at
        newly_attempted = self.attempted - self.initial_attempted
        if elapsed <= 0 or newly_attempted <= 0:
            return 0.0
        return newly_attempted / elapsed

    def snapshot(self, current_item: str | None = None) -> dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self.started_at
            remaining_seconds = self._remaining_seconds()
            rate = self._rate()
            return {
                "stage": self.stage,
                "total": self.total,
                "attempted": self.attempted,
                "completed": self.completed,
                "pending": max(self.total - self.completed, 0),
                "errors": self.errors,
                "elapsed_seconds": round(elapsed, 2),
                "eta_seconds": None if remaining_seconds is None else round(remaining_seconds, 2),
                "rate_items_per_second": round(rate, 4),
                "started_at": datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
                "last_updated": now_iso(),
                "current_item": current_item,
            }

    def _render(self, current_item: str | None = None) -> str:
        ratio = 0.0 if self.total == 0 else self.attempted / self.total
        filled = int(self.bar_width * ratio)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        remaining_seconds = self._remaining_seconds()
        rate = self._rate()
        item_suffix = f" current={current_item}" if current_item else ""
        return (
            f"[{self.stage}] {bar} {self.attempted}/{self.total} "
            f"({ratio * 100:5.1f}%) elapsed={format_seconds(time.time() - self.started_at)} "
            f"eta={format_seconds(remaining_seconds)} rate={rate:.2f}it/s ok={self.completed} err={self.errors}{item_suffix}"
        )

    def _emit(self, message: str, final: bool = False) -> None:
        if self._interactive:
            end = "\n" if final else ""
            sys.stdout.write("\r" + message + end)
            sys.stdout.flush()
        else:
            print(message, flush=True)

    def advance(self, current_item: str | None = None) -> None:
        with self._lock:
            self.attempted += 1
            self.completed += 1
            message = self._render(current_item=current_item)
        self._emit(message)

    def mark_error(self, current_item: str | None = None) -> None:
        with self._lock:
            self.attempted += 1
            self.errors += 1
            message = self._render(current_item=current_item)
        self._emit(message)

    def finish(self) -> None:
        with self._lock:
            message = self._render()
        self._emit(message, final=True)


def strip_markdown_fences(text: str) -> str:
    stripped = re.sub(r"```[a-zA-Z0-9_-]*", "", text)
    stripped = stripped.replace("```", "")
    return stripped.strip()


def normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


def parse_labeled_fields(text: str) -> dict[str, str]:
    cleaned = strip_markdown_fences(text)
    pattern = re.compile(
        r"(?ms)^\s*([A-Za-z][A-Za-z ]+)\s*:\s*(.*?)\s*(?=^\s*[A-Za-z][A-Za-z ]+\s*:|\Z)"
    )
    parsed: dict[str, str] = {}
    for raw_key, raw_value in pattern.findall(cleaned):
        key = raw_key.strip().lower().replace(" ", "_")
        parsed[key] = normalize_space(raw_value)
    return parsed


def first_text_line(text: str) -> str:
    cleaned = strip_markdown_fences(text)
    for line in cleaned.splitlines():
        line = normalize_space(line)
        if line:
            return line
    return ""


def normalize_visibility(value: str | None) -> str:
    text = (value or "").strip().lower()
    if any(token in text for token in ["uncertain", "not visible", "cannot", "unable", "insufficient", "unclear"]):
        return "uncertain"
    if any(token in text for token in ["medium", "partial", "partially", "limited"]):
        return "medium"
    if any(token in text for token in ["high", "clear", "clearly", "visible"]):
        return "high"
    if not text:
        return "uncertain"
    return "high"


def normalize_risk_direction(value: str | None) -> str:
    text = (value or "").strip().lower()
    if any(token in text for token in ["supports", "support", "higher risk", "increased risk", "consistent with risk"]):
        return "supports"
    if any(token in text for token in ["against", "low risk", "reassuring", "does not support", "not supportive"]):
        return "against"
    return "uncertain"


def normalize_confidence(value: str | None, visibility: str | None = None) -> str:
    text = (value or "").strip().lower()
    if "high" in text:
        return "high"
    if "medium" in text or "moderate" in text:
        return "medium"
    if "low" in text:
        return "low"
    if visibility == "uncertain":
        return "low"
    return "medium"


def normalize_evidence_strength(value: str | None, risk_direction: str, visibility: str | None = None) -> str:
    text = (value or "").strip().lower()
    if "strong" in text:
        return "strong"
    if "moderate" in text or "medium" in text:
        return "moderate"
    if "weak" in text or "mild" in text or "slight" in text:
        return "weak"
    if risk_direction == "uncertain":
        return "weak"
    if visibility == "uncertain":
        return "weak"
    return "moderate"


def parse_visual_response(raw_response: str, anatomy_target: str) -> dict[str, Any]:
    fields = parse_labeled_fields(raw_response)
    observation = fields.get("visualobservation") or fields.get("visual_observation") or first_text_line(raw_response)
    visibility = normalize_visibility(fields.get("visibility") or observation)
    parsed_target = fields.get("anatomytarget") or fields.get("anatomy_target") or anatomy_target
    parse_status = "structured" if ("visualobservation" in fields or "visual_observation" in fields) else "fallback"
    return {
        "anatomy_target": normalize_space(parsed_target) or anatomy_target,
        "visual_observation": observation or "No reliable observation extracted.",
        "visibility": visibility,
        "visual_parse_status": parse_status,
        "raw_visual_response": raw_response.strip(),
    }


def parse_evidence_card(
    raw_response: str,
    *,
    default_observation: str,
    default_visibility: str,
) -> tuple[dict[str, str], str]:
    fields = parse_labeled_fields(raw_response)
    thought = fields.get("thought", "")
    action = fields.get("action", "")
    observation = fields.get("observation") or default_observation
    interpretation = fields.get("interpretation") or observation
    final_thought = fields.get("finalthought") or fields.get("final_thought") or interpretation
    risk_direction = normalize_risk_direction(fields.get("riskdirection") or fields.get("risk_direction"))
    evidence_strength = normalize_evidence_strength(
        fields.get("evidencestrength") or fields.get("evidence_strength"),
        risk_direction=risk_direction,
        visibility=default_visibility,
    )
    confidence = normalize_confidence(
        fields.get("confidence"),
        visibility=default_visibility,
    )
    evidence_summary = fields.get("evidencesummary") or fields.get("evidence_summary") or interpretation
    parse_status = (
        "structured"
        if (
            ("thought" in fields or "observation" in fields or "interpretation" in fields or "finalthought" in fields)
            and risk_direction in {"supports", "against", "uncertain"}
        )
        else "fallback"
    )
    card = {
        "thought": thought or "Focused on the screening relevance of the observed anatomy.",
        "action": action or "Evaluate the screening relevance of the observed anatomy in the current clinical context.",
        "observation": observation or default_observation,
        "interpretation": interpretation or default_observation,
        "final_thought": final_thought or interpretation or default_observation,
        "risk_direction": risk_direction,
        "evidence_strength": evidence_strength,
        "confidence": confidence,
        "evidence_summary": evidence_summary or default_observation,
    }
    return card, parse_status


def flatten_evidence_cards(session: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(sorted(session, key=lambda value: int(value.get("session_index", 0))), start=1):
        evidence_card = item.get("evidence_card", {})
        lines.append(
            " | ".join(
                [
                    f"Session {index}",
                    f"Target={item.get('anatomy_target', '')}",
                    f"Visibility={item.get('visibility', '')}",
                    f"RiskDirection={evidence_card.get('risk_direction', '')}",
                    f"EvidenceStrength={evidence_card.get('evidence_strength', '')}",
                    f"Confidence={evidence_card.get('confidence', '')}",
                    f"Observation={evidence_card.get('observation') or item.get('visual_observation', '')}",
                    f"EvidenceSummary={evidence_card.get('evidence_summary', '')}",
                ]
            )
        )
    return "\n".join(lines)


def extract_binary_answer(text: str) -> tuple[str, str]:
    final_answer_match = re.search(r"final answer\s*:\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if final_answer_match:
        return final_answer_match.group(1).lower(), "final_answer"

    matches = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower(), "fallback_last_token"

    return "unknown", "unparsed"


def build_output_paths(output_dir: str | Path) -> dict[str, str]:
    root = Path(output_dir)
    return {
        "visual_jsonl": str(root / "visual.jsonl"),
        "visual_state": str(root / "visual.state.json"),
        "visual_errors": str(root / "visual.errors.jsonl"),
        "reason_jsonl": str(root / "reason.jsonl"),
        "reason_state": str(root / "reason.state.json"),
        "reason_errors": str(root / "reason.errors.jsonl"),
        "final_jsonl": str(root / "final.jsonl"),
        "final_csv": str(root / "final.csv"),
        "final_state": str(root / "final.state.json"),
        "final_errors": str(root / "final.errors.jsonl"),
        "metrics_json": str(root / "metrics.json"),
        "analysis_json": str(root / "analysis.json"),
        "run_manifest": str(root / "run.json"),
    }


def enforce_output_policy(paths: dict[str, str], resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")

    target_files = list(paths.values())
    if overwrite:
        for path in target_files:
            remove_file(path)
        return

    if resume:
        return

    existing = [path for path in target_files if Path(path).exists()]
    if existing:
        joined = ", ".join(existing)
        raise FileExistsError(f"Outputs already exist: {joined}. Use --resume or --overwrite.")


def write_stage_state(
    state_path: str,
    tracker: ProgressTracker,
    *,
    status: str,
    output_path: str,
    error_path: str,
    current_item: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = tracker.snapshot(current_item=current_item)
    payload.update(
        {
            "status": status,
            "output_path": output_path,
            "error_path": error_path,
        }
    )
    if extra:
        payload.update(extra)
    write_json_atomic(payload, state_path)


def write_run_manifest(
    manifest_path: str,
    *,
    script_name: str,
    args: dict[str, Any],
    resolved_models: dict[str, str],
    output_dir: str,
) -> None:
    write_json_atomic(
        {
            "script": script_name,
            "created_at": now_iso(),
            "arguments": args,
            "resolved_models": resolved_models,
            "output_dir": output_dir,
        },
        manifest_path,
    )


def load_existing_stage_records(path: str, target_paths: set[str], resume: bool) -> list[dict[str, Any]]:
    if not resume or not Path(path).exists():
        return []
    return [record for record in read_jsonl(path, strict=False) if record.get("image_path") in target_paths]


def sync_final_csv_from_jsonl(records: list[dict[str, Any]], csv_path: str) -> None:
    rows = []
    for record in sort_records_by_sample_index(records):
        rows.append(
            {
                "image_path": record["image_path"],
                "gold_label": record["binary_label"],
                "predicted_label": record.get("predicted_label", "unknown"),
                "match": record.get("match", False),
                "parse_status": record.get("parse_status", "unparsed"),
                "final_response": record.get("final_response", ""),
            }
        )
    write_csv(rows, csv_path, fieldnames=FINAL_CSV_FIELDS)


def to_error_record(stage: str, image_path: str, exc: Exception) -> dict[str, Any]:
    return {
        "stage": stage,
        "image_path": image_path,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "timestamp": now_iso(),
    }


def load_prediction_records(pred_file: str | Path) -> list[dict[str, Any]]:
    path = Path(pred_file)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path, strict=False)
    if path.suffix.lower() == ".csv":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    {
                        "image_path": row.get("image_path", ""),
                        "binary_label": row.get("gold_label", ""),
                        "predicted_label": row.get("predicted_label", "unknown"),
                        "match": row.get("match", ""),
                        "parse_status": row.get("parse_status", "unparsed"),
                        "final_response": row.get("final_response", ""),
                    }
                )
        return records
    raise ValueError(f"Unsupported prediction file format: {path}")


def compute_binary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_records = [record for record in records if record.get("binary_label") in {"yes", "no"}]
    tp = tn = fp = fn = parsed = unknown = 0
    for record in valid_records:
        gold = record["binary_label"]
        predicted = str(record.get("predicted_label", "unknown")).lower()
        if predicted not in {"yes", "no"}:
            predicted = "unknown"
        if predicted == "unknown":
            unknown += 1
        else:
            parsed += 1

        if gold == "yes":
            if predicted == "yes":
                tp += 1
            else:
                fn += 1
        else:
            if predicted == "no":
                tn += 1
            else:
                fp += 1

    total = len(valid_records)
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1_score = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )
    return {
        "total": total,
        "valid": total,
        "parsed": parsed,
        "unknown": unknown,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score,
        "report": {
            "Accuracy (%)": round(accuracy * 100, 4),
            "Sensitivity (%)": round(sensitivity * 100, 4),
            "Specificity (%)": round(specificity * 100, 4),
            "F1-Score (%)": round(f1_score * 100, 4),
        },
    }


class QwenReasonerRunner:
    def __init__(
        self,
        model_name: str | None = None,
        settings: GenerationSettings | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        self.model_name = resolve_model_source(
            explicit_path=model_name,
            env_var_name="QWEN_REASONER_MODEL_PATH",
            default_hf_model=DEFAULT_REASONER_HF_MODEL,
            name_hints=["Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen2.5-3B-Instruct", "Qwen2.5-3B"],
        )
        self.settings = settings or GenerationSettings()
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_count = 0

    def _lazy_load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        with self._load_lock:
            if self._tokenizer is not None and self._model is not None:
                return

            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            configure_left_padding(self._tokenizer, self._model)
            configure_generation_config(
                self._model,
                do_sample=self.settings.temperature > 0,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                repetition_penalty=self.settings.repetition_penalty,
            )
            self._load_count += 1

    @property
    def device(self):
        self._lazy_load()
        return next(self._model.parameters()).device

    @property
    def load_count(self) -> int:
        return self._load_count

    def ensure_loaded(self) -> None:
        self._lazy_load()

    def close(self) -> None:
        self._tokenizer = None
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _build_messages(self, prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _generate_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.settings.max_new_tokens,
            "repetition_penalty": self.settings.repetition_penalty,
            "do_sample": self.settings.temperature > 0,
        }
        if self.settings.temperature > 0:
            kwargs["temperature"] = self.settings.temperature
            kwargs["top_p"] = self.settings.top_p
        return kwargs

    def generate_batch(self, prompts: list[str], system_prompt: str | None = None) -> list[str]:
        if not prompts:
            return []
        self._lazy_load()

        with self._infer_lock:
            texts = [
                self._tokenizer.apply_chat_template(
                    self._build_messages(prompt, system_prompt=system_prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

            import torch

            model_inputs = self._tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated_ids = self._model.generate(**model_inputs, **self._generate_kwargs())
            trimmed_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = [text.strip() for text in self._tokenizer.batch_decode(trimmed_ids, skip_special_tokens=True)]
        ensure_batch_alignment(responses, len(prompts), "QwenReasonerRunner.generate_batch")
        return responses

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.generate_batch([prompt], system_prompt=system_prompt)[0]


class QwenVLRunner:
    def __init__(
        self,
        model_name: str | None = None,
        settings: GenerationSettings | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        self.model_name = resolve_model_source(
            explicit_path=model_name,
            env_var_name="QWEN_VL_MODEL_PATH",
            default_hf_model=DEFAULT_VL_HF_MODEL,
            name_hints=[
                "Qwen2.5-VL-7B-Instruct",
                "Qwen2.5-VL-7B",
                "Qwen2.5-VL-3B-Instruct",
                "Qwen2.5-VL-3B",
            ],
        )
        self.settings = settings or GenerationSettings()
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._processor = None
        self._model = None
        self._process_vision_info = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_count = 0

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None and self._process_vision_info is not None:
            return
        with self._load_lock:
            if self._processor is not None and self._model is not None and self._process_vision_info is not None:
                return

            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                raise ImportError(
                    "qwen-vl-utils is required for Qwen2.5-VL inference. "
                    "Install it before running EviOSAHS experiments."
                ) from exc

            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            if hasattr(self._processor, "tokenizer"):
                configure_left_padding(self._processor.tokenizer, self._model)
            configure_generation_config(
                self._model,
                do_sample=self.settings.temperature > 0,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                repetition_penalty=self.settings.repetition_penalty,
            )
            if hasattr(self._processor, "padding_side"):
                self._processor.padding_side = "left"
            self._process_vision_info = process_vision_info
            self._load_count += 1

    @property
    def device(self):
        self._lazy_load()
        return next(self._model.parameters()).device

    @property
    def load_count(self) -> int:
        return self._load_count

    def ensure_loaded(self) -> None:
        self._lazy_load()

    def close(self) -> None:
        self._processor = None
        self._model = None
        self._process_vision_info = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _build_messages(
        self,
        image_path: str | Path,
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        image_uri = Path(image_path).resolve().as_uri()
        user_content = [
            {"type": "image", "image": image_uri},
            {"type": "text", "text": prompt},
        ]
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_text_messages(self, prompt: str, system_prompt: str | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _generate_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.settings.max_new_tokens,
            "repetition_penalty": self.settings.repetition_penalty,
            "do_sample": self.settings.temperature > 0,
        }
        if self.settings.temperature > 0:
            kwargs["temperature"] = self.settings.temperature
            kwargs["top_p"] = self.settings.top_p
        return kwargs

    def generate_batch(
        self,
        image_paths: list[str | Path],
        prompts: list[str],
        system_prompt: str | None = None,
    ) -> list[str]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length.")
        if not image_paths:
            return []
        self._lazy_load()

        with self._infer_lock:
            message_batch = [
                self._build_messages(image_path=image_path, prompt=prompt, system_prompt=system_prompt)
                for image_path, prompt in zip(image_paths, prompts)
            ]
            texts = [
                self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in message_batch
            ]
            image_inputs, video_inputs = self._process_vision_info(message_batch)
            inputs = self._processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            import torch

            with torch.inference_mode():
                generated_ids = self._model.generate(**inputs, **self._generate_kwargs())
            trimmed_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = [text.strip() for text in self._processor.batch_decode(trimmed_ids, skip_special_tokens=True)]
        ensure_batch_alignment(responses, len(prompts), "QwenVLRunner.generate_batch")
        return responses

    def generate(self, image_path: str | Path, prompt: str, system_prompt: str | None = None) -> str:
        return self.generate_batch([image_path], [prompt], system_prompt=system_prompt)[0]

    def generate_text_batch(self, prompts: list[str], system_prompt: str | None = None) -> list[str]:
        if not prompts:
            return []
        self._lazy_load()

        with self._infer_lock:
            message_batch = [
                self._build_text_messages(prompt=prompt, system_prompt=system_prompt)
                for prompt in prompts
            ]
            texts = [
                self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in message_batch
            ]
            inputs = self._processor(text=texts, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)

            import torch

            with torch.inference_mode():
                generated_ids = self._model.generate(**inputs, **self._generate_kwargs())
            trimmed_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = [text.strip() for text in self._processor.batch_decode(trimmed_ids, skip_special_tokens=True)]
        ensure_batch_alignment(responses, len(prompts), "QwenVLRunner.generate_text_batch")
        return responses

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.generate_text_batch([prompt], system_prompt=system_prompt)[0]


class Llava16VisualRunner:
    def __init__(
        self,
        model_name: str | None = None,
        settings: GenerationSettings | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        self.model_name = resolve_model_source(
            explicit_path=model_name,
            env_var_name="LLAVA16_MODEL_PATH",
            default_hf_model=DEFAULT_LLAVA16_HF_MODEL,
            name_hints=[
                "llava-v1.6-mistral-7b-hf",
                "llava1.6",
                "llava-1.6",
                "llava",
            ],
        )
        self.settings = settings or GenerationSettings()
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._processor = None
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_count = 0

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        with self._load_lock:
            if self._processor is not None and self._model is not None:
                return

            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

            self._processor = LlavaNextProcessor.from_pretrained(self.model_name)
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            if hasattr(self._processor, "tokenizer"):
                configure_left_padding(self._processor.tokenizer, self._model)
            configure_generation_config(
                self._model,
                do_sample=self.settings.temperature > 0,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                repetition_penalty=self.settings.repetition_penalty,
            )
            if hasattr(self._processor, "padding_side"):
                self._processor.padding_side = "left"
            self._load_count += 1

    @property
    def device(self):
        self._lazy_load()
        configured_device = first_device_from_map(self.device_map)
        if configured_device is not None:
            import torch

            return torch.device(configured_device)
        return next(self._model.parameters()).device

    @property
    def load_count(self) -> int:
        return self._load_count

    def ensure_loaded(self) -> None:
        self._lazy_load()

    def close(self) -> None:
        self._processor = None
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _format_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        merged_prompt = prompt.strip()
        if system_prompt:
            merged_prompt = f"{system_prompt.strip()}\n\n{merged_prompt}"
        return f"[INST] <image>\n{merged_prompt} [/INST]"

    def _generate_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": self.settings.max_new_tokens,
            "repetition_penalty": self.settings.repetition_penalty,
            "do_sample": self.settings.temperature > 0,
        }
        if self.settings.temperature > 0:
            kwargs["temperature"] = self.settings.temperature
            kwargs["top_p"] = self.settings.top_p
        return kwargs

    def generate_batch(
        self,
        image_paths: list[str | Path],
        prompts: list[str],
        system_prompt: str | None = None,
    ) -> list[str]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length.")
        if not image_paths:
            return []
        self._lazy_load()

        from PIL import Image
        import torch

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        with self._infer_lock:
            texts = [self._format_prompt(prompt, system_prompt=system_prompt) for prompt in prompts]
            inputs = self._processor(text=texts, images=images, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                generated_ids = self._model.generate(**inputs, **self._generate_kwargs())
            trimmed_ids = [
                output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            responses = [text.strip() for text in self._processor.batch_decode(trimmed_ids, skip_special_tokens=True)]
        ensure_batch_alignment(responses, len(prompts), "Llava16VisualRunner.generate_batch")
        return responses

    def generate(self, image_path: str | Path, prompt: str, system_prompt: str | None = None) -> str:
        return self.generate_batch([image_path], [prompt], system_prompt=system_prompt)[0]
