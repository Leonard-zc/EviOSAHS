from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from sklearn.model_selection import StratifiedKFold

from eviosahs.models import (
    STRUCTURED_FEATURE_NAMES,
    binary_label_from_numeric,
    build_clinical_summary_from_structured_text,
    build_structured_feature_dict,
    ensure_parent_dir,
    load_pickle_dataset,
    now_iso,
    resolve_image_path,
    write_json_atomic,
    write_jsonl,
)


MANIFEST_CSV_FIELDS = [
    "sample_index",
    "patient_id",
    "image_path",
    "numeric_label",
    "binary_label",
    "semantic_text",
    "clinical_summary_clean",
    *STRUCTURED_FEATURE_NAMES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare EviOSAHS manifests and CV folds.")
    parser.add_argument("--input-pkl", required=True, help="Path to the local cohort pickle")
    parser.add_argument("--image-dir", required=True, help="Directory containing patient images")
    parser.add_argument("--output-dir", required=True, help="Output directory for manifests and folds")
    parser.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fold generation")
    return parser.parse_args()


def build_manifest_records(dataset: list[dict[str, Any]], image_dir: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(dataset):
        image_name = f"{sample['id']}.jpg"
        image_path = resolve_image_path(image_dir, image_name)
        feature_dict = build_structured_feature_dict(sample.get("text"))
        record = {
            "sample_index": sample_index,
            "patient_id": str(sample["id"]),
            "image_path": str(image_path),
            "image_name": image_name,
            "numeric_label": int(sample["label"]),
            "binary_label": binary_label_from_numeric(sample["label"]),
            "semantic_text": sample.get("semantic_text", ""),
            "clinical_summary_clean": build_clinical_summary_from_structured_text(sample.get("text")),
            "structured_text": sample.get("text", []),
            "structured_features": feature_dict,
            "created_at": now_iso(),
        }
        record.update(feature_dict)
        records.append(record)
    return records


def write_manifest_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in MANIFEST_CSV_FIELDS})


def build_folds(records: list[dict[str, Any]], folds: int, seed: int) -> list[dict[str, Any]]:
    labels = [record["binary_label"] for record in records]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_payloads: list[dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(records, labels)):
        train_records = [records[index] for index in train_idx]
        test_records = [records[index] for index in test_idx]
        fold_payloads.append(
            {
                "fold_index": fold_index,
                "train_indices": [int(index) for index in train_idx],
                "test_indices": [int(index) for index in test_idx],
                "train_image_names": [record["image_name"] for record in train_records],
                "test_image_names": [record["image_name"] for record in test_records],
                "train_size": len(train_records),
                "test_size": len(test_records),
                "train_positive": sum(1 for record in train_records if record["binary_label"] == "yes"),
                "test_positive": sum(1 for record in test_records if record["binary_label"] == "yes"),
            }
        )
    return fold_payloads


def write_fold_manifests(records: list[dict[str, Any]], fold_payloads: list[dict[str, Any]], output_dir: str | Path) -> None:
    root = Path(output_dir)
    for fold in fold_payloads:
        fold_root = root / "folds" / f"fold_{fold['fold_index']}"
        train_records = [records[index] for index in fold["train_indices"]]
        test_records = [records[index] for index in fold["test_indices"]]
        write_jsonl(train_records, fold_root / "train.jsonl")
        write_jsonl(test_records, fold_root / "test.jsonl")
        write_manifest_csv(train_records, fold_root / "train.csv")
        write_manifest_csv(test_records, fold_root / "test.csv")
        write_json_atomic(fold, fold_root / "fold_meta.json")


def main() -> None:
    args = parse_args()
    dataset = load_pickle_dataset(args.input_pkl)
    records = build_manifest_records(dataset, args.image_dir)
    folds = build_folds(records, args.folds, args.seed)

    output_dir = Path(args.output_dir)
    manifest_jsonl = output_dir / "manifest.jsonl"
    manifest_csv = output_dir / "manifest.csv"
    folds_json = output_dir / "folds.json"
    dataset_info_json = output_dir / "dataset_info.json"

    write_jsonl(records, manifest_jsonl)
    write_manifest_csv(records, manifest_csv)
    write_json_atomic(
        {
            "created_at": now_iso(),
            "input_pkl": str(Path(args.input_pkl).resolve()),
            "image_dir": str(Path(args.image_dir).resolve()),
            "total_samples": len(records),
            "folds": args.folds,
            "seed": args.seed,
            "binary_label_counts": {
                "yes": sum(1 for record in records if record["binary_label"] == "yes"),
                "no": sum(1 for record in records if record["binary_label"] == "no"),
            },
            "numeric_label_counts": {
                str(label): sum(1 for record in records if int(record["numeric_label"]) == label)
                for label in sorted({int(record["numeric_label"]) for record in records})
            },
        },
        dataset_info_json,
    )
    write_json_atomic({"folds": folds}, folds_json)
    write_fold_manifests(records, folds, output_dir)

    print(f"Prepared manifest: {manifest_jsonl}")
    print(f"Prepared folds: {folds_json}")


if __name__ == "__main__":
    main()
