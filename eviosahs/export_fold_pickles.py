from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from eviosahs.models import ensure_parent_dir, load_pickle_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fold-specific pickle subsets for zero-shot baselines.")
    parser.add_argument("--input-pkl", required=True, help="Path to the local cohort pickle")
    parser.add_argument("--folds-json", required=True, help="folds.json from prepare_dataset.py")
    parser.add_argument("--output-dir", required=True, help="Output directory for fold subset pickles")
    return parser.parse_args()


def build_lookup(dataset: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {f"{sample['id']}.jpg": sample for sample in dataset}


def dump_pickle(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("wb") as handle:
        pickle.dump(records, handle)


def main() -> None:
    args = parse_args()
    dataset = load_pickle_dataset(args.input_pkl)
    lookup = build_lookup(dataset)
    folds = json.loads(Path(args.folds_json).read_text(encoding="utf-8"))["folds"]
    output_root = Path(args.output_dir)

    for fold in folds:
        fold_root = output_root / f"fold_{fold['fold_index']}"
        train_records = [lookup[image_name] for image_name in fold["train_image_names"] if image_name in lookup]
        test_records = [lookup[image_name] for image_name in fold["test_image_names"] if image_name in lookup]
        dump_pickle(train_records, fold_root / "train_subset.pkl")
        dump_pickle(test_records, fold_root / "test_subset.pkl")

    print(f"Exported fold pickles to: {output_root}")


if __name__ == "__main__":
    main()
