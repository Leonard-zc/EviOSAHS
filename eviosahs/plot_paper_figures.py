from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-eviosahs")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory


METHOD_LABELS = {
    "T5": "Clinical-only",
    "F1": "Direct InstructBLIP",
    "F2": "Direct LLaVA-1.6",
    "F3": "Direct Qwen2.5-VL",
    "F4": "Naive two-stage",
    "F5": "EviOSAHS",
    "F6": "Early clinical fusion",
    "S1": "Single-model Qwen",
}

MAIN_ORDER = ["F5", "F6", "S1", "T5", "F2", "F4", "F3", "F1"]
PAIR_ORDER = ["T5", "F2", "F4", "F6", "S1"]

BACKBONE_LABELS = {
    "F5": "Qwen-VL + Qwen reasoner\n(EviOSAHS)",
    "X1": "LLaVA-1.6 visual\n+ Llama reasoner",
    "X2": "Qwen-VL visual\n+ Llama reasoner",
    "X3": "LLaVA-1.6 visual\n+ Qwen reasoner",
}

ABLATION_LABELS = {
    "F5": "Full workflow",
    "F5_A2": "Remove ReAct-style\nevidence reasoning",
    "F5_A3": "Remove structured\nclinical summary",
    "F5_A4": "Remove evidence\nstrength grading",
    "F5_A5": "Remove balanced\nfinal adjudication",
    "F5_A6": "Single-pass visual\nextraction",
}

CASE_COMPARATOR_LABELS = {
    "T5_clinical_only": "Clinical-only",
    "F2_llava16_direct": "Direct LLaVA-1.6",
    "F4_two_stage_naive": "Naive two-stage",
    "F5_A6_single_pass": "Single-pass visual",
    "S1_single_model_balanced": "Single-model Qwen",
}

ANATOMY_LABELS = {
    "neck": "Neck",
    "mouth": "Mouth",
    "face_and_neck_fat": "Face/neck fat",
    "lower_jaw": "Lower jaw",
    "chin": "Chin",
    "midface": "Midface",
    "nose": "Nose",
    "profile": "Profile",
}

COLORS = {
    "blue": "#2166AC",
    "blue_dark": "#0F4D92",
    "blue_light": "#92C5DE",
    "blue_pale": "#D1E5F0",
    "red": "#B2182B",
    "red_dark": "#8E1021",
    "red_light": "#F4A3A3",
    "red_pale": "#FDDBC7",
    "orange": "#B2182B",
    "green": "#2166AC",
    "purple": "#92C5DE",
    "amber": "#E6A700",
    "dark": "#202020",
    "text": "#2F2F2F",
    "muted": "#6F6F6F",
    "grey": "#AFAFAF",
    "light": "#E6E6E6",
    "grid": "#E9E9E9",
    "lighter": "#F7F7F7",
    "white": "#FFFFFF",
}

ACTIVE_FIGURE_NAMES = (
    "fig4_main_prediction_behavior",
    "fig5_visual_audit_image_controls",
    "fig6_subgroup_error_attribution",
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.8,
            "axes.titlesize": 7.0,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "legend.fontsize": 5.9,
            "legend.frameon": False,
            "legend.handlelength": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COLORS["dark"],
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.tiff", bbox_inches="tight", dpi=600)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=360)
    plt.close(fig)


def clean_axis(ax: plt.Axes, grid_axis: str | None = "x") -> None:
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.45)
        ax.set_axisbelow(True)


def emphasize_tick(ax: plt.Axes, text: str, color: str | None = None) -> None:
    for label in ax.get_yticklabels():
        if text in label.get_text():
            label.set_fontweight("bold")
            if color:
                label.set_color(color)


def parse_ci(ci: str | float | int | None) -> tuple[float, float]:
    if ci is None or pd.isna(ci):
        return (np.nan, np.nan)
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(ci))
    if len(nums) < 2:
        return (np.nan, np.nan)
    return (float(nums[0]), float(nums[1]))


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return ""
    if p == 0 or p < 1e-6:
        return "p < 1e-6"
    return f"p = {p:.3f}"


def wrap(s: str | None, width: int) -> str:
    if s is None:
        return ""
    s = " ".join(str(s).split())
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False))


def sanitize_text(s: str | None) -> str:
    if not s:
        return ""
    out = str(s)
    replacements = {
        "F5_A6": "single-pass variant",
        "F5": "EviOSAHS",
        "T5": "clinical-only baseline",
        "F2": "LLaVA direct baseline",
        "F4": "naive two-stage baseline",
        "S1": "single-model Qwen variant",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def read_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "ablation": pd.read_csv(results_dir / "figure2_ablation_delta.csv"),
        "main": pd.read_csv(results_dir / "table2_main_binary_results_with_ci.csv"),
        "paired": pd.read_csv(results_dir / "table3_paired_comparisons_vs_f5.csv"),
        "distribution": pd.read_csv(results_dir / "figure3_prediction_distribution.csv"),
        "visual_quality": pd.read_csv(results_dir / "figure4a_visual_quality_by_question.csv"),
        "proxy": pd.read_csv(results_dir / "figure4b_visual_proxy_agreement.csv"),
        "v6_metrics": pd.read_csv(results_dir / "figure5_v6_control_metrics.csv"),
        "v6_flips": pd.read_csv(results_dir / "figure5_v6_flip_counts.csv"),
        "backbone": pd.read_csv(results_dir / "table_backbone_transfer.csv"),
        "subgroup": pd.read_csv(results_dir / "table_f5_subgroup_analysis.csv"),
    }


def plot_main_behavior(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    main = data["main"].copy()
    main["label"] = main["id"].map(METHOD_LABELS)
    main = main.set_index("id").loc[MAIN_ORDER].reset_index()

    paired = data["paired"].query("group == 'main'").copy()
    paired = paired[paired["id"].isin(PAIR_ORDER)]
    paired["label"] = paired["id"].map(METHOD_LABELS)
    paired = paired.set_index("id").loc[PAIR_ORDER].reset_index()

    paired["net_gain"] = paired["f5_correct_method_wrong"] - paired["f5_wrong_method_correct"]
    paired = paired.sort_values("net_gain", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(7.35, 4.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    y_pair = np.arange(len(paired))
    evi_only = paired["f5_correct_method_wrong"].to_numpy()
    comparator_only = paired["f5_wrong_method_correct"].to_numpy()
    net_gain = paired["net_gain"].to_numpy()
    net_col = max(evi_only) + 36
    p_col = max(evi_only) + 88
    ax_a.hlines(y_pair, comparator_only, evi_only, color=COLORS["light"], linewidth=6.4, zorder=1)
    ax_a.scatter(comparator_only, y_pair, s=42, color=COLORS["red"], edgecolor="white", linewidth=0.7, zorder=3)
    ax_a.scatter(evi_only, y_pair, s=58, color=COLORS["blue"], edgecolor="white", linewidth=0.7, zorder=4)
    for i, row in paired.iterrows():
        ax_a.text(row["f5_wrong_method_correct"] - 5.0, i, f"{int(row['f5_wrong_method_correct'])}", ha="right", va="center", fontsize=6.0, color=COLORS["red"], fontweight="bold")
        ax_a.text(row["f5_correct_method_wrong"] + 5.0, i, f"{int(row['f5_correct_method_wrong'])}", ha="left", va="center", fontsize=6.0, color=COLORS["blue"], fontweight="bold")
        ax_a.text(net_col, i, f"+{int(row['net_gain'])}", ha="left", va="center", fontsize=6.5, color=COLORS["dark"], fontweight="bold")
        ax_a.text(p_col, i, fmt_p(row["mcnemar_exact_p"]), ha="left", va="center", fontsize=5.7, color=COLORS["muted"])
    ax_a.set_yticks(y_pair, paired["label"])
    ax_a.invert_yaxis()
    ax_a.set_ylim(len(paired) - 0.5, -0.78)
    ax_a.set_xlim(0, max(evi_only) + 136)
    ax_a.set_xlabel("Discordant subjects")
    ax_a.set_title("Paired correctness advantage", pad=8)
    ax_a.text(18, -0.54, "comparator only", color=COLORS["red"], fontsize=5.9, va="bottom", fontweight="bold")
    ax_a.text(148, -0.54, "EviOSAHS only", color=COLORS["blue"], fontsize=5.9, va="bottom", fontweight="bold")
    ax_a.text(net_col, -0.54, "net", color=COLORS["dark"], fontsize=5.7, va="bottom", fontweight="bold")
    ax_a.text(p_col, -0.54, "McNemar", color=COLORS["muted"], fontsize=5.7, va="bottom", ha="left", fontweight="bold")
    clean_axis(ax_a, "x")
    add_panel_label(ax_a, "a")

    y = np.arange(len(main))
    pred_yes = main["pred_yes"] / main["n"] * 100
    pred_no = main["pred_no"] / main["n"] * 100
    pred_unknown = main["pred_unknown"] / main["n"] * 100
    ax_b.barh(y, pred_yes, color=COLORS["blue"], height=0.52, label="Positive")
    ax_b.barh(y, pred_no, left=pred_yes, color=COLORS["red"], height=0.52, label="Negative")
    ax_b.barh(y, pred_unknown, left=pred_yes + pred_no, color="#858585", height=0.52, label="Unknown")
    ax_b.set_yticks(y, main["label"])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Predictions (%)")
    ax_b.set_title("Output distribution", pad=7)
    ax_b.tick_params(axis="y", labelsize=5.8, pad=2)
    emphasize_tick(ax_b, "EviOSAHS", COLORS["blue"])
    clean_axis(ax_b, "x")
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=5.8, handlelength=0.95, columnspacing=0.60)
    add_panel_label(ax_b, "b")

    ax_c.set_xlim(-0.48, 1.48)
    ax_c.set_ylim(len(main) - 0.5, -0.5)
    for sep in np.arange(0.5, len(main), 1):
        ax_c.axhline(sep, color="#F0F0F0", linewidth=0.5, zorder=0)
    for i, row in main.iterrows():
        for x_pos, col, color in [(0, "fn", COLORS["red"]), (1, "fp", COLORS["blue"])]:
            count = int(row[col])
            size = 13 + np.sqrt(count) * 5.2
            ax_c.scatter(x_pos, i, s=size, color=color, edgecolor="white", linewidth=0.65, zorder=3)
            text_x = x_pos - 0.08 if col == "fn" else x_pos + 0.08
            text_ha = "right" if col == "fn" else "left"
            ax_c.text(text_x, i, f"{count}", ha=text_ha, va="center", fontsize=5.8, color=color, fontweight="bold", zorder=4)
    ax_c.set_xticks([0, 1], ["FN", "FP"])
    error_labels = [x.replace("Early clinical fusion", "Clinical fusion").replace("Direct ", "") for x in main["label"]]
    ax_c.set_yticks(np.arange(len(main)), error_labels)
    ax_c.tick_params(axis="y", labelsize=5.5, pad=1)
    emphasize_tick(ax_c, "EviOSAHS", COLORS["blue"])
    ax_c.tick_params(axis="x", length=0, pad=2)
    ax_c.set_title("Error burden", pad=7)
    ax_c.text(0.50, -0.14, "area ~ count", transform=ax_c.transAxes, ha="center", va="top", fontsize=5.5, color=COLORS["muted"])
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.grid(False)
    add_panel_label(ax_c, "c")

    save_figure(fig, out_dir, "fig4_main_prediction_behavior")


def plot_component_ablation(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    ab = data["ablation"].copy()
    row_order = ["F5", "F5_A2", "F5_A3", "F5_A4", "F5_A5", "F5_A6"]
    ab = ab[ab["id"].isin(row_order)].set_index("id").loc[row_order].reset_index()
    ab["label"] = ab["id"].map(ABLATION_LABELS)

    components = [
        "Seven-question\nvisual",
        "ReAct evidence\nreasoning",
        "Structured\nclinical text",
        "Strength\ngrading",
        "Balanced\nadjudication",
    ]
    matrix = np.ones((len(ab), len(components)))
    absent = {
        "F5_A2": [1],
        "F5_A3": [2],
        "F5_A4": [3],
        "F5_A5": [4],
        "F5_A6": [0],
    }
    for i, row in ab.iterrows():
        for j in absent.get(row["id"], []):
            matrix[i, j] = 0

    fig = plt.figure(figsize=(7.55, 4.05), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    ax_a.set_xlim(-0.5, len(components) - 0.5)
    ax_a.set_ylim(len(ab) - 0.5, -0.5)
    ax_a.set_xticks(np.arange(len(components)), components, rotation=35, ha="right")
    ax_a.set_yticks(np.arange(len(ab)), ab["label"])
    ax_a.set_title("Component map")
    ax_a.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_a.add_patch(
                Rectangle(
                    (j - 0.48, i - 0.48),
                    0.96,
                    0.96,
                    facecolor="#F7F7F7",
                    edgecolor="#E3E3E3",
                    linewidth=0.45,
                )
            )
            if matrix[i, j] == 1:
                ax_a.scatter(j, i, s=44, color=COLORS["blue_light"], edgecolor=COLORS["blue"], linewidth=0.8, zorder=3)
            else:
                ax_a.scatter(j, i, s=62, marker="X", color=COLORS["orange"], edgecolor="white", linewidth=0.5, zorder=4)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.grid(False)
    add_panel_label(ax_a, "a")

    rows = ab[ab["id"] != "F5"].copy()
    y = np.arange(len(rows))
    sens_delta = rows["delta_vs_f5_sensitivity_pp"].to_numpy()
    acc_delta = rows["delta_vs_f5_accuracy_pp"].to_numpy()
    for i in range(len(rows)):
        ax_b.plot([sens_delta[i], acc_delta[i]], [i, i], color=COLORS["light"], linewidth=2.0, zorder=1)
    ax_b.scatter(sens_delta, y, color=COLORS["red"], s=32, label="Sensitivity", zorder=3, edgecolor="white", linewidth=0.6)
    ax_b.scatter(acc_delta, y, color=COLORS["blue"], s=32, label="Accuracy", zorder=3, edgecolor="white", linewidth=0.6)
    ax_b.axvline(0, color=COLORS["dark"], linewidth=0.7)
    ax_b.set_yticks(y, rows["label"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Change from full EviOSAHS (percentage points)")
    ax_b.set_title("Performance change after component removal")
    ax_b.set_xlim(min(sens_delta.min(), acc_delta.min()) - 4, max(3, sens_delta.max() + 3))
    clean_axis(ax_b, "x")
    ax_b.text(0.02, 0.96, "Sensitivity", transform=ax_b.transAxes, ha="left", va="top", fontsize=6.2, color=COLORS["red"])
    ax_b.text(0.02, 0.90, "Accuracy", transform=ax_b.transAxes, ha="left", va="top", fontsize=6.2, color=COLORS["blue"])
    add_panel_label(ax_b, "b")

    save_figure(fig, out_dir, "fig_component_ablation")


def plot_backbone_operating_points(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    main = data["main"].query("id == 'F5'").copy()
    main = main.assign(method="Qwen visual + Qwen final", id="F5")
    bb = data["backbone"].copy()
    bb = pd.concat(
        [
            main[["id", "method", "accuracy_percent", "sensitivity_percent", "specificity_percent", "f1_score_percent"]],
            bb[["id", "method", "accuracy_percent", "sensitivity_percent", "specificity_percent", "f1_score_percent"]],
        ],
        ignore_index=True,
    )
    bb["label"] = bb["id"].map(BACKBONE_LABELS)

    fig, ax = plt.subplots(figsize=(4.55, 3.8), constrained_layout=True)
    point_colors = {
        "F5": COLORS["blue"],
        "X1": "#A6A6A6",
        "X2": COLORS["purple"],
        "X3": COLORS["green"],
    }
    for _, row in bb.iterrows():
        ax.scatter(
            row["specificity_percent"],
            row["sensitivity_percent"],
            s=18 + row["f1_score_percent"] * 4.7,
            color=point_colors.get(row["id"], COLORS["grey"]),
            edgecolor="white",
            linewidth=0.75,
            alpha=0.96,
            zorder=3,
        )
    label_offsets = {
        "F5": (10, -28),
        "X1": (8, -2),
        "X2": (8, 3),
        "X3": (8, -3),
    }
    for _, row in bb.iterrows():
        dx, dy = label_offsets.get(row["id"], (5, 4))
        ax.annotate(
            row["label"],
            (row["specificity_percent"], row["sensitivity_percent"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.3,
            fontweight="bold" if row["id"] == "F5" else "normal",
            color=COLORS["blue"] if row["id"] == "F5" else COLORS["text"],
        )
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Specificity (%)")
    ax.set_ylabel("Sensitivity (%)")
    ax.set_title("Backbone-transfer operating points")
    ax.text(0.03, 0.04, "Point size: F1-score", transform=ax.transAxes, color=COLORS["muted"], fontsize=6.6)
    clean_axis(ax, "both")
    save_figure(fig, out_dir, "fig_backbone_operating_points")


def plot_visual_control(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    quality = data["visual_quality"].copy()
    quality["label"] = quality["anatomy_target"].map(ANATOMY_LABELS).fillna(quality["anatomy_target"])
    quality = quality.sort_values("visibility_uncertain_percent", ascending=False)
    v6 = data["v6_metrics"].copy()
    flips = data["v6_flips"].copy()

    fig = plt.figure(figsize=(7.35, 4.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.18], height_ratios=[1.08, 0.92])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    y = np.arange(len(quality))
    unc = quality["visibility_uncertain_percent"].to_numpy()
    ax_a.hlines(y, 0, unc, color=COLORS["red_pale"], linewidth=5.0, zorder=1)
    ax_a.scatter(unc, y, s=46, color=COLORS["red"], edgecolor="white", linewidth=0.7, zorder=3)
    for i, value in enumerate(unc):
        label = f"{value:.1f}" if value > 0 else "0"
        ax_a.text(value + 0.7, i, label, ha="left", va="center", fontsize=6.2, color=COLORS["red"] if value > 0 else COLORS["muted"], fontweight="bold" if value > 0 else "normal")
    ax_a.set_yticks(y, quality["label"])
    ax_a.invert_yaxis()
    emphasize_tick(ax_a, "Nose", COLORS["red"])
    ax_a.set_xlim(0, 36)
    ax_a.set_xlabel("Uncertain visibility (%)")
    ax_a.set_title("Uncertain visual evidence is anatomically localized", pad=7)
    clean_axis(ax_a, "x")
    add_panel_label(ax_a, "a")

    controls = v6[v6["Setting"].isin(["image-shuffle", "blur"])].copy()
    metric_rows = [
        ("Delta Accuracy", "Accuracy"),
        ("Delta Sensitivity", "Sensitivity"),
        ("Delta F1", "F1-score"),
    ]
    y2 = np.arange(len(metric_rows))
    control_styles = {
        "image-shuffle": ("Image shuffle", COLORS["red"], -0.10),
        "blur": ("Gaussian blur", COLORS["blue"], 0.10),
    }
    for _, row in controls.iterrows():
        label, color, offset = control_styles[row["Setting"]]
        vals = [row[col] for col, _ in metric_rows]
        for yi, val in zip(y2, vals):
            ax_b.plot([0, val], [yi + offset, yi + offset], color=color, linewidth=1.25, alpha=0.95)
            ax_b.scatter(val, yi + offset, s=22, color=color, edgecolor="white", linewidth=0.6, zorder=3, label=label if yi == 0 else None)
            ax_b.text(val - 0.42 if val < 0 else val + 0.42, yi + offset, f"{val:.1f}", ha="right" if val < 0 else "left", va="center", fontsize=5.8, color=color, fontweight="bold")
    ax_b.axvline(0, color=COLORS["dark"], linewidth=0.7)
    ax_b.set_yticks(y2, [label for _, label in metric_rows])
    ax_b.invert_yaxis()
    ax_b.set_ylim(len(metric_rows) - 0.05, -0.55)
    ax_b.set_xlim(-17.2, 21.5)
    ax_b.set_xlabel("Delta vs original EviOSAHS (pp)")
    ax_b.set_title("Image perturbation controls", pad=7)
    clean_axis(ax_b, "x")
    ax_b.legend(loc="lower left", handlelength=0.9)
    add_panel_label(ax_b, "b")

    flips = flips.copy()
    flips["label"] = flips["setting"].str.replace(" seed42", "", regex=False).str.replace(" radius12", "", regex=False)
    flip_labels = flips["label"].str.replace("Image shuffle", "Image\nshuffle").str.replace("Gaussian blur", "Gaussian\nblur").to_list()
    loss = flips["f5_correct_control_wrong"].to_numpy()
    recovery = flips["f5_wrong_control_correct"].to_numpy()
    y3 = np.arange(len(flips))
    ax_c.axvline(0, color=COLORS["dark"], linewidth=0.7)
    ax_c.barh(y3, -loss, color=COLORS["blue"], height=0.46, label="Original only")
    ax_c.barh(y3, recovery, color=COLORS["red"], height=0.46, label="Control only")
    for i, row in flips.iterrows():
        ax_c.text(-row["f5_correct_control_wrong"] - 1.6, i, f"{int(row['f5_correct_control_wrong'])}", ha="right", va="center", fontsize=5.9, color=COLORS["blue"], fontweight="bold")
        ax_c.text(row["f5_wrong_control_correct"] + 1.6, i, f"{int(row['f5_wrong_control_correct'])}", ha="left", va="center", fontsize=5.9, color=COLORS["red"], fontweight="bold")
        ax_c.text(42.0, i, f"{row['changed_percent']:.1f}% changed", ha="left", va="center", fontsize=5.7, color=COLORS["muted"])
    ax_c.set_yticks(y3, flip_labels)
    ax_c.invert_yaxis()
    ax_c.set_ylim(len(flips) - 0.35, -0.35)
    ax_c.set_xlim(-46, 58)
    ax_c.set_xlabel("Prediction flips")
    ax_c.set_title("Flip direction", pad=7)
    ax_c.text(0.02, 1.02, "original-only correct", transform=ax_c.transAxes, ha="left", va="bottom", fontsize=5.7, color=COLORS["blue"], fontweight="bold")
    ax_c.text(0.98, 1.02, "control-only correct", transform=ax_c.transAxes, ha="right", va="bottom", fontsize=5.7, color=COLORS["red"], fontweight="bold")
    clean_axis(ax_c, "x")
    add_panel_label(ax_c, "c")

    save_figure(fig, out_dir, "fig5_visual_audit_image_controls")


def subgroup_rows(subgroup: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        ("sex", "female", "Female", "Sex"),
        ("sex", "male", "Male", "Sex"),
        ("age_group", "<35", "Age <35", "Age"),
        ("age_group", "35-49", "Age 35-49", "Age"),
        ("age_group", ">=50", "Age >=50", "Age"),
        ("bmi_category", "healthy_weight", "Healthy weight", "BMI"),
        ("bmi_category", "overweight", "Overweight", "BMI"),
        ("bmi_category", "obesity", "Obesity", "BMI"),
        ("whr_category", "not_elevated", "WHR not elevated", "WHR"),
        ("whr_category", "borderline_high", "WHR borderline high", "WHR"),
        ("whr_category", "elevated", "WHR elevated", "WHR"),
        ("whr_category", "markedly_elevated", "WHR markedly elevated", "WHR"),
    ]
    out = []
    for var, level, label, group in wanted:
        row = subgroup[(subgroup["variable"] == var) & (subgroup["level"] == level)]
        if row.empty:
            continue
        r = row.iloc[0].copy()
        r["display_label"] = label
        r["display_group"] = group
        r["fnr_percent"] = 100 - float(r["sensitivity_percent"])
        out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)


def compute_evidence_error_counts(results_dir: Path) -> pd.DataFrame:
    with open(results_dir / "figure7_case_evidence_traces.json", "r", encoding="utf-8") as f:
        json.load(f)
    final_path = results_dir.parent / "outputs/run_matrix/main/F5_qwen_two_stage_final_only_clinical/final.jsonl"
    groups = {"False positive": [], "False negative": [], "True positive": [], "True negative": []}
    with open(final_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold = row.get("binary_label")
            pred = row.get("predicted_label")
            if gold == "no" and pred == "yes":
                key = "False positive"
            elif gold == "yes" and pred == "no":
                key = "False negative"
            elif gold == "yes" and pred == "yes":
                key = "True positive"
            elif gold == "no" and pred == "no":
                key = "True negative"
            else:
                continue
            counts = {"supports": 0, "against": 0, "uncertain": 0}
            for session in row.get("session", []):
                direction = session.get("evidence_card", {}).get("risk_direction", "uncertain")
                if direction in counts:
                    counts[direction] += 1
                else:
                    counts["uncertain"] += 1
            groups[key].append(counts)
    rows = []
    for key, arr in groups.items():
        n = len(arr)
        rows.append(
            {
                "group": key,
                "n": n,
                "supports": np.mean([x["supports"] for x in arr]) if n else 0,
                "against": np.mean([x["against"] for x in arr]) if n else 0,
                "uncertain": np.mean([x["uncertain"] for x in arr]) if n else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_subgroup_error(data: dict[str, pd.DataFrame], results_dir: Path, out_dir: Path) -> None:
    sub = subgroup_rows(data["subgroup"])
    evidence = compute_evidence_error_counts(results_dir)
    evidence = evidence.set_index("group").loc[["False positive", "False negative"]].reset_index()

    fig = plt.figure(figsize=(7.35, 3.72), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.018, wspace=0.04, hspace=0.02)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    group_colors = {"Sex": COLORS["blue"], "Age": COLORS["blue_light"], "BMI": COLORS["red"], "WHR": COLORS["red_light"]}
    y = np.arange(len(sub))
    sizes = 18 + np.sqrt(sub["n"].to_numpy()) * 2.1
    band_colors = {"Sex": "#F3F7FB", "Age": "#F6FAFC", "BMI": "#FFF4F3", "WHR": "#FFF7F7"}
    group_ranges = []
    for group, rows in sub.groupby("display_group", sort=False):
        group_ranges.append((group, rows.index.min(), rows.index.max()))
    for group, lo, hi in group_ranges:
        ax_a.axhspan(lo - 0.5, hi + 0.5, color=band_colors[group], zorder=-2)
    for sep in [1.5, 4.5, 7.5]:
        ax_a.axhline(sep, color="#DCDCDC", linewidth=0.65, zorder=0)
    for i, row in sub.iterrows():
        color = group_colors[row["display_group"]]
        ax_a.plot([0, row["fnr_percent"]], [i, i], color="#D8D8D8", linewidth=1.0, zorder=1)
        ax_a.scatter(row["fnr_percent"], i, color=color, s=sizes[i], zorder=3, edgecolor="white", linewidth=0.65)
        ax_a.text(21.8, i, f"n={int(row['n'])}", va="center", ha="right", fontsize=5.8, color=COLORS["muted"])
    overall = data["main"].query("id == 'F5'").iloc[0]
    overall_fnr = overall["fn"] / (overall["tp"] + overall["fn"]) * 100
    ax_a.axvline(overall_fnr, color=COLORS["dark"], linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax_a.text(overall_fnr + 0.3, -0.72, f"overall {overall_fnr:.1f}%", fontsize=5.9, color=COLORS["dark"], ha="left", va="top", fontweight="bold")
    ax_a.set_yticks(y, sub["display_label"])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("False-negative rate (%)")
    ax_a.set_title("Subgroup missed-case profile", pad=7)
    ax_a.set_xlim(-0.65, 22.4)
    clean_axis(ax_a, "x")
    handles = [plt.Line2D([0], [0], marker="o", color=v, label=k, linestyle="", markersize=5) for k, v in group_colors.items()]
    ax_a.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.145), ncol=4, handletextpad=0.35, columnspacing=0.75, borderaxespad=0.0)
    add_panel_label(ax_a, "a")

    evidence_types = [
        ("supports", "Supporting", COLORS["blue"]),
        ("against", "Opposing", COLORS["red"]),
        ("uncertain", "Uncertain", "#E9ECEF"),
    ]
    x = np.arange(len(evidence))
    bottom = np.zeros(len(evidence))
    for col, label, color in evidence_types:
        vals = evidence[col].to_numpy()
        bars = ax_b.bar(x, vals, bottom=bottom, width=0.54, color=color, label=label, edgecolor="white", linewidth=0.6)
        for xi, val, btm, bar in zip(x, vals, bottom, bars):
            if val >= 0.55:
                text_color = "white" if col in {"supports", "against"} else COLORS["dark"]
                ax_b.text(xi, btm + val / 2, f"{val:.1f}", ha="center", va="center", fontsize=6.0, color=text_color, fontweight="bold")
        bottom += vals
    for xi, total in zip(x, bottom):
        ax_b.text(xi, total + 0.16, f"{total:.1f}", ha="center", va="bottom", fontsize=6.2, color=COLORS["dark"], fontweight="bold")
    ax_b.set_xticks(x, [f"{row['group']}\n(n={int(row['n'])})" for _, row in evidence.iterrows()])
    ax_b.set_xlim(-0.34, 1.34)
    ax_b.set_ylim(0, 7.45)
    ax_b.set_ylabel("Mean number of evidence cards")
    ax_b.set_title("Evidence-card composition in errors", pad=7)
    clean_axis(ax_b, "y")
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, -0.145), ncol=3, fontsize=5.8, handlelength=0.9, columnspacing=0.7, borderaxespad=0.0)
    add_panel_label(ax_b, "b")

    save_figure(fig, out_dir, "fig6_subgroup_error_attribution")


def draw_badge(ax: plt.Axes, x: float, y: float, text: str, color: str, width: float = 0.07) -> None:
    patch = FancyBboxPatch(
        (x, y - 0.014),
        width,
        0.028,
        boxstyle="round,pad=0.003,rounding_size=0.006",
        facecolor=color,
        edgecolor="none",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=6.0, color="white", transform=ax.transAxes)


def draw_trace_section(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str, mono: bool = False) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.007",
            facecolor=COLORS["lighter"],
            edgecolor="#D6D6D6",
            linewidth=0.55,
            transform=ax.transAxes,
        )
    )
    ax.text(x + 0.01, y + h - 0.018, title, fontsize=7.6, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(
        x + 0.01,
        y + h - 0.043,
        body,
        fontsize=5.9,
        transform=ax.transAxes,
        va="top",
        family="DejaVu Sans Mono" if mono else "DejaVu Sans",
        color=COLORS["text"],
    )


def draw_case_page(fig: plt.Figure, case: dict) -> None:
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    trace = case["evi_osahs_full_trace"]
    inp = trace["stage_0_case_input"]
    final = trace["stage_3_final_adjudication_full"]
    parsed = final.get("parsed_final_response", {})
    sessions = trace["stage_1_visual_and_stage_2_evidence_sessions_full"]
    counts = trace["evidence_card_counts"]

    pred_color = COLORS["green"] if case["evi_osahs_match"] else COLORS["red"]
    ax.add_patch(Rectangle((0.025, 0.925), 0.95, 0.055, facecolor=pred_color, alpha=0.09, edgecolor="none", transform=ax.transAxes))
    ax.text(
        0.04,
        0.958,
        f"{sanitize_text(case['role'])} | sample {case['sample_index']} | gold={case['gold_binary']} | EviOSAHS={case['evi_osahs_prediction']}",
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="center",
        transform=ax.transAxes,
        color=COLORS["dark"],
    )
    ax.text(0.04, 0.93, wrap(sanitize_text(case["purpose"]), 120), fontsize=6.6, ha="left", va="top", transform=ax.transAxes, color=COLORS["muted"])

    clinical = inp.get("clinical_summary", "").replace("PatientProfile:\n", "")
    clinical = "\n".join(clinical.splitlines()[:8])
    draw_trace_section(
        ax,
        (0.04, 0.755),
        (0.38, 0.145),
        "Stage 0. Structured input",
        f"Image: {inp.get('image_path')}\nOriginal AHI label: {inp.get('numeric_label')}\n{clinical}",
        mono=True,
    )

    ax.text(0.455, 0.895, "Comparator outputs", fontsize=7.8, fontweight="bold", transform=ax.transAxes)
    comp = case.get("comparator_outputs_full", {})
    for i, (name, item) in enumerate(comp.items()):
        yy = 0.867 - i * 0.03
        label = CASE_COMPARATOR_LABELS.get(name, sanitize_text(name).replace("_", " "))
        pred = (item or {}).get("predicted_label", "NA")
        color = COLORS["green"] if (item or {}).get("match") else COLORS["red"]
        ax.text(0.455, yy, label, fontsize=6.5, transform=ax.transAxes, va="center", color=COLORS["text"])
        draw_badge(ax, 0.735, yy, str(pred), color, width=0.048)

    ax.text(0.04, 0.72, "Stages 1-2. Anatomy-specific visual evidence", fontsize=8.0, fontweight="bold", transform=ax.transAxes)
    cols = [0.04, 0.14, 0.34, 0.58, 0.84, 0.91]
    headers = ["Target", "Visual observation", "Evidence interpretation", "Evidence summary", "Strength", "Direction"]
    for x, h in zip(cols, headers):
        ax.text(x, 0.696, h, fontsize=6.2, fontweight="bold", transform=ax.transAxes, color=COLORS["muted"])

    row_top = 0.678
    row_h = 0.071
    direction_colors = {"supports": COLORS["blue"], "against": COLORS["red"], "uncertain": "#8E8E8E"}
    for i, s in enumerate(sessions):
        yrow = row_top - i * row_h
        bg = "#FFFFFF" if i % 2 == 0 else "#F7F7F7"
        ax.add_patch(Rectangle((0.035, yrow - row_h + 0.006), 0.93, row_h - 0.008, facecolor=bg, edgecolor="#E2E2E2", linewidth=0.35, transform=ax.transAxes))
        card = s.get("evidence_card", {})
        direction = card.get("risk_direction", "uncertain")
        strength = card.get("evidence_strength", "")
        target = ANATOMY_LABELS.get(s.get("anatomy_target", ""), s.get("anatomy_target", ""))
        ax.text(cols[0], yrow - 0.014, target, fontsize=6.2, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(cols[1], yrow - 0.010, wrap(s.get("visual_observation", ""), 31), fontsize=5.45, transform=ax.transAxes, va="top")
        ax.text(cols[2], yrow - 0.010, wrap(card.get("interpretation", ""), 38), fontsize=5.45, transform=ax.transAxes, va="top")
        ax.text(cols[3], yrow - 0.010, wrap(card.get("evidence_summary", ""), 38), fontsize=5.45, transform=ax.transAxes, va="top")
        draw_badge(ax, cols[4], yrow - 0.024, strength[:3] if strength else "NA", COLORS["amber"], width=0.045)
        draw_badge(ax, cols[5], yrow - 0.024, direction[:4], direction_colors.get(direction, COLORS["grey"]), width=0.048)

    final_text = sanitize_text(final.get("final_response", ""))
    draw_trace_section(ax, (0.04, 0.04), (0.61, 0.125), "Stage 3. Final adjudication response", wrap(final_text, 132), mono=True)
    ax.text(0.70, 0.145, "Evidence balance", fontsize=7.8, fontweight="bold", transform=ax.transAxes)
    draw_badge(ax, 0.70, 0.113, f"S {counts.get('supports', 0)}", COLORS["blue"], width=0.058)
    draw_badge(ax, 0.765, 0.113, f"A {counts.get('against', 0)}", COLORS["red"], width=0.058)
    draw_badge(ax, 0.83, 0.113, f"U {counts.get('uncertain', 0)}", "#8E8E8E", width=0.058)
    risk = parsed.get("clinical_risk_level", "")
    ans = parsed.get("final_answer", final.get("predicted_label", ""))
    ax.text(0.70, 0.074, f"Clinical risk: {risk}", fontsize=7.0, transform=ax.transAxes)
    ax.text(0.70, 0.048, f"Final answer: {ans}", fontsize=8.8, fontweight="bold", transform=ax.transAxes, color=pred_color)


def plot_evidence_traces(results_dir: Path, out_dir: Path) -> None:
    with open(results_dir / "figure7_case_evidence_traces.json", "r", encoding="utf-8") as f:
        traces = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig_evidence_traces.pdf"
    with PdfPages(pdf_path) as pdf:
        for case in traces["cases"]:
            fig = plt.figure(figsize=(12.8, 8.0), constrained_layout=False)
            draw_case_page(fig, case)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    first = traces["cases"][0]
    fig = plt.figure(figsize=(12.8, 8.0), constrained_layout=False)
    draw_case_page(fig, first)
    fig.savefig(out_dir / "fig_evidence_traces_page1.png", bbox_inches="tight", dpi=360)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Nature-style figures for the EviOSAHS paper.")
    parser.add_argument("--results-dir", default="results", help="Directory containing locally generated result CSV/JSON files.")
    parser.add_argument("--output-dir", default="results/figures", help="Output directory for generated figures.")
    args = parser.parse_args()

    set_style()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    data = read_results(results_dir)

    # Keep the manuscript figure set intentionally narrow: three quantitative pages.
    plot_main_behavior(data, out_dir)
    plot_visual_control(data, out_dir)
    plot_subgroup_error(data, results_dir, out_dir)

    print(f"Wrote {len(ACTIVE_FIGURE_NAMES)} figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
