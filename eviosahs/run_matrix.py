from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from eviosahs.experiment_registry import ExperimentSpec, experiment_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or execute the full EviOSAHS experiment matrix.")
    parser.add_argument("--output-root", default="outputs/run_matrix", help="Root directory for all experiment outputs")
    parser.add_argument("--input-pkl", required=True, help="Path to the local cohort pickle")
    parser.add_argument("--image-dir", required=True, help="Directory containing patient images")
    parser.add_argument("--qwen-root", default=None, help="Directory containing Qwen model weights")
    parser.add_argument("--instructblip-model-path", default=None, help="Optional local path for InstructBLIP weights")
    parser.add_argument("--llava-model-path", default=None, help="Optional local path for LLaVA-1.6 weights")
    parser.add_argument("--llama-model-path", default=None, help="Optional local path for Llama-3.1-8B-Instruct weights")
    parser.add_argument("--python-bin", default="python", help="Python executable")
    parser.add_argument(
        "--group",
        choices=["preparation", "main", "ablation", "backbone", "all"],
        default="all",
        help="Which experiment group to print or execute",
    )
    parser.add_argument("--experiment-id", action="append", default=[], help="Only run selected experiment ids")
    parser.add_argument("--vl-device", default="cuda:0", help="VL device for Qwen two-stage runs")
    parser.add_argument("--reasoner-device", default="cuda:1", help="Reasoner device for Qwen two-stage runs")
    parser.add_argument("--device", default="cuda:0", help="Single-device baselines device")
    parser.add_argument("--resume", action="store_true", help="Resume selected experiments when outputs already exist.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite selected experiments when outputs already exist.")
    parser.add_argument("--mode", choices=["print", "execute"], default="print", help="Print commands or execute them")
    return parser.parse_args()


def select_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs = experiment_specs(args.output_root)
    if args.group == "all":
        specs = [spec for spec in specs if spec.group != "backbone"]
    else:
        specs = [spec for spec in specs if spec.group == args.group]
    if args.experiment_id:
        wanted = set(args.experiment_id)
        specs = [spec for spec in experiment_specs(args.output_root) if spec.experiment_id in wanted]
    return specs


def build_command(spec: ExperimentSpec, args: argparse.Namespace) -> list[str]:
    module_name = Path(spec.script).with_suffix("").name
    command = [args.python_bin, "-m", f"eviosahs.{module_name}"]

    if spec.script == "prepare_dataset.py":
        command.extend(["--input-pkl", args.input_pkl, "--image-dir", args.image_dir])
    elif spec.script == "classical_baselines.py":
        pass
    elif spec.script == "foundation_baselines.py":
        command.extend(["--input-pkl", args.input_pkl, "--image-dir", args.image_dir, "--device", args.device])
        if args.resume:
            command.append("--resume")
        if args.overwrite:
            command.append("--overwrite")
        if args.qwen_root:
            command.extend(
                [
                    "--qwen-vl-model-path",
                    str(Path(args.qwen_root) / "Qwen2.5-VL-7B-Instruct"),
                    "--qwen-text-model-path",
                    str(Path(args.qwen_root) / "Qwen2.5-7B-Instruct"),
                ]
            )
        if args.instructblip_model_path:
            command.extend(["--instructblip-model-path", args.instructblip_model_path])
        if args.llava_model_path:
            command.extend(["--llava-model-path", args.llava_model_path])
    elif spec.script == "two_stage_binary.py":
        command.extend(
            [
                "--input-pkl",
                args.input_pkl,
                "--image-dir",
                args.image_dir,
                "--vl-device",
                args.vl_device,
                "--reasoner-device",
                args.reasoner_device,
                "--continue-on-error",
            ]
        )
        if spec.group != "backbone":
            command.extend(
                [
                    "--visual-batch-size",
                    "7",
                    "--reason-batch-size",
                    "8",
                    "--final-batch-size",
                    "8",
                ]
            )
        if args.resume:
            command.append("--resume")
        if args.overwrite:
            command.append("--overwrite")
        if spec.group == "backbone":
            if spec.experiment_id == "X1_llava16_llama31_two_stage":
                if not args.llava_model_path:
                    raise ValueError("X1 requires --llava-model-path.")
                if not args.llama_model_path:
                    raise ValueError("X1 requires --llama-model-path.")
                command.extend(
                    [
                        "--vl-runner-type",
                        "llava16",
                        "--vl-model-path",
                        args.llava_model_path,
                        "--reasoner-model-path",
                        args.llama_model_path,
                    ]
                )
            elif spec.experiment_id == "X2_qwenvl_llama31_two_stage":
                if not args.qwen_root:
                    raise ValueError("X2 requires --qwen-root.")
                if not args.llama_model_path:
                    raise ValueError("X2 requires --llama-model-path.")
                command.extend(
                    [
                        "--vl-runner-type",
                        "qwen",
                        "--vl-model-path",
                        str(Path(args.qwen_root) / "Qwen2.5-VL-7B-Instruct"),
                        "--reasoner-model-path",
                        args.llama_model_path,
                    ]
                )
            elif spec.experiment_id == "X3_llava16_qwen7b_two_stage":
                if not args.llava_model_path:
                    raise ValueError("X3 requires --llava-model-path.")
                if not args.qwen_root:
                    raise ValueError("X3 requires --qwen-root.")
                command.extend(
                    [
                        "--vl-runner-type",
                        "llava16",
                        "--vl-model-path",
                        args.llava_model_path,
                        "--reasoner-model-path",
                        str(Path(args.qwen_root) / "Qwen2.5-7B-Instruct"),
                    ]
                )
            else:
                raise ValueError(f"Unsupported backbone transfer experiment: {spec.experiment_id}")
        elif args.qwen_root:
            command.extend(
                [
                    "--vl-model-path",
                    str(Path(args.qwen_root) / "Qwen2.5-VL-7B-Instruct"),
                    "--reasoner-model-path",
                    str(Path(args.qwen_root) / "Qwen2.5-7B-Instruct"),
                ]
            )
    else:
        raise ValueError(f"Unsupported script in registry: {spec.script}")

    command.extend([part for part in spec.arguments if part != ""])
    return command


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive.")
    specs = select_specs(args)
    if not specs:
        raise SystemExit("No experiments selected.")

    for spec in specs:
        command = build_command(spec, args)
        print(f"[{spec.group}] {spec.experiment_id} :: {spec.description}")
        print(shlex.join(command))
        print()
        if args.mode == "execute":
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
