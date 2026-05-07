from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from eviosahs.models import load_pickle_dataset, now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic V6 image-control directories.")
    parser.add_argument("--input-pkl", required=True, help="Dataset pickle used by the F5/V6 runs.")
    parser.add_argument("--image-dir", required=True, help="Original image directory.")
    parser.add_argument("--output-root", default="outputs/image_controls", help="Root for control image directories.")
    parser.add_argument("--mode", choices=["image-shuffle", "blur", "blank", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic image shuffling.")
    parser.add_argument("--blur-radius", type=float, default=12.0, help="Gaussian blur radius for the blur control.")
    parser.add_argument("--blank-color", default="127,127,127", help="RGB color for blank image control.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing control directories.")
    return parser.parse_args()


def image_names_from_dataset(input_pkl: str | Path) -> list[str]:
    dataset = load_pickle_dataset(input_pkl)
    names = [f"{sample['id']}.jpg" for sample in dataset]
    return sorted(dict.fromkeys(names))


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError("--blank-color must be R,G,B with values in [0, 255].")
    return tuple(parts)  # type: ignore[return-value]


def copy_shuffled_images(image_dir: Path, names: list[str], output_dir: Path, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    if len(names) <= 1:
        shuffled = names[:]
    else:
        shuffled = names[:]
        for _ in range(1000):
            rng.shuffle(shuffled)
            if all(src != dst for src, dst in zip(shuffled, names)):
                break
        else:
            shuffled = names[1:] + names[:1]

    mapping: list[dict[str, str]] = []
    unchanged = 0
    for dst_name, src_name in zip(names, shuffled):
        if dst_name == src_name:
            unchanged += 1
        shutil.copy2(image_dir / src_name, output_dir / dst_name)
        mapping.append({"target_image": dst_name, "source_image": src_name})
    return {"mapping": mapping, "unchanged_count": unchanged}


def write_blurred_images(image_dir: Path, names: list[str], output_dir: Path, radius: float) -> None:
    for name in names:
        with Image.open(image_dir / name) as image:
            blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
            blurred.save(output_dir / name, quality=95)


def write_blank_images(image_dir: Path, names: list[str], output_dir: Path, color: tuple[int, int, int]) -> None:
    for name in names:
        with Image.open(image_dir / name) as image:
            blank = Image.new("RGB", image.size, color=color)
            blank.save(output_dir / name, quality=95)


def validate_inputs(image_dir: Path, names: list[str]) -> None:
    missing = [name for name in names if not (image_dir / name).exists()]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} images under {image_dir}: {preview}")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_existing_modes(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    modes = payload.get("modes", {})
    return modes if isinstance(modes, dict) else {}


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_root = Path(args.output_root)
    names = image_names_from_dataset(args.input_pkl)
    validate_inputs(image_dir, names)

    requested_modes = ["image-shuffle", "blur", "blank"] if args.mode == "all" else [args.mode]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created_at": now_iso(),
        "input_pkl": str(args.input_pkl),
        "image_dir": str(image_dir),
        "output_root": str(output_root),
        "sample_count": len(names),
        "modes": load_existing_modes(manifest_path),
    }

    if "image-shuffle" in requested_modes:
        out_dir = output_root / "image_shuffle_seed42"
        prepare_dir(out_dir, args.overwrite)
        shuffle_payload = copy_shuffled_images(image_dir, names, out_dir, args.seed)
        manifest["modes"]["image-shuffle"] = {
            "output_dir": str(out_dir),
            "seed": args.seed,
            "unchanged_count": shuffle_payload["unchanged_count"],
            "mapping_file": str(out_dir / "shuffle_mapping.json"),
        }
        write_manifest(
            out_dir / "shuffle_mapping.json",
            {
                "created_at": now_iso(),
                "seed": args.seed,
                "target_keeps_original_filename": True,
                "mapping": shuffle_payload["mapping"],
            },
        )

    if "blur" in requested_modes:
        out_dir = output_root / f"blur_radius{args.blur_radius:g}"
        prepare_dir(out_dir, args.overwrite)
        write_blurred_images(image_dir, names, out_dir, args.blur_radius)
        manifest["modes"]["blur"] = {
            "output_dir": str(out_dir),
            "blur_radius": args.blur_radius,
        }

    if "blank" in requested_modes:
        out_dir = output_root / "blank_gray"
        prepare_dir(out_dir, args.overwrite)
        color = parse_rgb(args.blank_color)
        write_blank_images(image_dir, names, out_dir, color)
        manifest["modes"]["blank"] = {
            "output_dir": str(out_dir),
            "blank_color": color,
        }

    write_manifest(manifest_path, manifest)
    print(f"Wrote V6 controls for {len(names)} samples to {output_root}")
    for mode, details in manifest["modes"].items():
        print(f"{mode}: {details['output_dir']}")


if __name__ == "__main__":
    main()
