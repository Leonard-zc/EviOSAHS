# EviOSAHS

Official code for the manuscript **Structured Visual Evidence Decomposition for Evidence-Grounded Multimodal Screening of Obstructive Sleep Apnea-Hypopnea Syndrome**.

**EviOSAHS** is a staged multimodal screening pipeline for obstructive sleep apnea-hypopnea syndrome (OSAHS). It converts a patient facial/neck image into anatomy-specific visual observations, summarizes those observations as structured evidence cards, and combines them with a clean clinical summary for final binary screening adjudication.

This repository contains code only. It does not include patient data, model weights, generated predictions, metric tables, manuscript figures, or case-study outputs.

## Repository Structure

```text
EviOSAHS/
├── eviosahs/                 # Python source code
├── scripts/                  # Reproduction command wrappers
├── data/                     # Local data placeholder; raw data are not distributed
├── models/                   # Local model-weight placeholder; weights are not distributed
├── outputs/                  # Local raw experiment outputs
├── results/                  # Local derived tables and figures
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── CITATION.cff
└── README.md
```

Only this root `README.md` is used for documentation. Local outputs under `outputs/` and `results/` are ignored by git.

## Method Correspondence

The code follows the paper workflow:

| Paper component | Code location |
|---|---|
| Structured clinical-summary reconstruction | `eviosahs/models.py` |
| Seven anatomy-specific visual questions | `eviosahs/prompts.py` |
| Visual observation stage | `eviosahs/two_stage_binary.py` |
| Evidence-card reasoning | `eviosahs/prompts.py`, `eviosahs/models.py` |
| Final-only clinical adjudication | `eviosahs/two_stage_binary.py` |
| Clinical-only and direct MLLM baselines | `eviosahs/foundation_baselines.py` |
| F5 component ablations | `eviosahs/experiment_registry.py` |
| Backbone-transfer experiments | `eviosahs/experiment_registry.py` |
| Visual-output audit and proxy checks | `eviosahs/visual_eval_*.py` |
| Image-control analysis | `eviosahs/visual_eval_v6_*.py` |
| Binary evaluation | `eviosahs/eval_binary.py` |
| Figure generation from local result tables | `eviosahs/plot_paper_figures.py` |

## Installation

Use Python 3.10+.

```bash
cd EviOSAHS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Editable installation is optional:

```bash
pip install -e .
```

## Required Local Assets

The following assets must be prepared outside the repository:

| Argument | Description |
|---|---|
| `--input-pkl` | Local cohort pickle containing image IDs, labels, clinical fields, and optional semantic text |
| `--image-dir` | Directory containing patient images referenced by the cohort file |
| `--qwen-root` | Directory containing `Qwen2.5-VL-7B-Instruct` and `Qwen2.5-7B-Instruct` |
| `--llava-model-path` | LLaVA-1.6 model path or Hugging Face identifier |
| `--llama-model-path` | Llama-3.1-8B-Instruct model path |
| `--instructblip-model-path` | InstructBLIP model path or Hugging Face identifier |

The primary binary label mapping used by the code is:

```text
0 -> screening-negative
1/2/3 -> screening-positive
```

## Quick Start

Print the registered main experiments without running inference:

```bash
python -m eviosahs.run_matrix \
  --input-pkl /path/to/cohort.pkl \
  --image-dir /path/to/images \
  --qwen-root /path/to/qwen_weights \
  --group main \
  --mode print
```

Run the primary EviOSAHS method:

```bash
python -m eviosahs.run_matrix \
  --input-pkl /path/to/cohort.pkl \
  --image-dir /path/to/images \
  --qwen-root /path/to/qwen_weights \
  --experiment-id F5_qwen_two_stage_final_only_clinical \
  --output-root outputs/run_matrix \
  --mode execute
```

The same command can be launched through the shell wrapper:

```bash
INPUT_PKL=/path/to/cohort.pkl \
IMAGE_DIR=/path/to/images \
QWEN_ROOT=/path/to/qwen_weights \
bash scripts/run_primary_f5.sh
```

## Experiment IDs

Experiments are registered in `eviosahs/experiment_registry.py`.

| Experiment ID | Paper role |
|---|---|
| `T5_clinical_only_qwen_text` | Clinical-only LLM baseline |
| `F1_instructblip_direct` | InstructBLIP direct multimodal baseline |
| `F2_llava16_direct` | LLaVA-1.6 direct multimodal baseline |
| `F3_qwen_direct` | Qwen2.5-VL direct multimodal baseline |
| `F4_qwen_two_stage_naive` | Naive two-stage Qwen baseline |
| `F5_qwen_two_stage_final_only_clinical` | Primary EviOSAHS method |
| `F6_ours_eviosahs_balanced` | Clinical-in-reason variant |
| `F5_A2_wo_react` | F5 ablation without ReAct-style evidence reasoning |
| `F5_A3_wo_structured_clinical` | F5 ablation without structured clinical summary |
| `F5_A4_wo_evidence_strength` | F5 ablation without evidence-strength grading |
| `F5_A5_wo_balanced_final` | F5 ablation without balanced final adjudication |
| `F5_A6_wo_7q_decomposition` | F5 ablation without seven-question visual decomposition |
| `X1_llava16_llama31_two_stage` | Backbone transfer: LLaVA visual + Llama final |
| `X2_qwenvl_llama31_two_stage` | Backbone transfer: Qwen-VL visual + Llama final |
| `X3_llava16_qwen7b_two_stage` | Backbone transfer: LLaVA visual + Qwen final |

## Reproduction Wrappers

The scripts in `scripts/` are thin wrappers around `python -m eviosahs...`. They use environment variables so private data and model paths stay outside the repository.

| Script | Purpose |
|---|---|
| `scripts/print_experiment_matrix.sh` | Print commands for a selected experiment group |
| `scripts/run_primary_f5.sh` | Run the primary EviOSAHS/F5 pipeline |
| `scripts/run_main_baselines.sh` | Run the main clinical-only and direct multimodal baselines |
| `scripts/run_f5_ablations.sh` | Run F5-centered ablations |
| `scripts/run_backbone_transfer.sh` | Run backbone-transfer experiments |
| `scripts/build_visual_audit.sh` | Build visual-output quality and proxy-consistency artifacts from completed F5 outputs |
| `scripts/make_image_controls.sh` | Create deterministic image-control directories |
| `scripts/make_paper_figures.sh` | Generate manuscript figures from locally generated result tables |

Example:

```bash
INPUT_PKL=/path/to/cohort.pkl \
IMAGE_DIR=/path/to/images \
QWEN_ROOT=/path/to/qwen_weights \
RESUME=1 \
bash scripts/run_primary_f5.sh
```

## Output Convention

Generated files should stay local:

| Directory | Contents |
|---|---|
| `outputs/run_matrix/` | Raw experiment outputs such as `visual.jsonl`, `reason.jsonl`, `final.jsonl`, run manifests, and intermediate states |
| `results/metrics/` | Metrics JSON files generated from local predictions |
| `results/tables/` | Locally collated manuscript tables |
| `results/figures/` | Locally generated manuscript figures |
| `results/visual_audit/` | Visual quality, proxy-consistency, and image-control summaries |

No generated result files are committed in this code release.

## Evaluation

Evaluate a locally generated prediction file:

```bash
python -m eviosahs.eval_binary \
  --pred-file outputs/run_matrix/main/F5_qwen_two_stage_final_only_clinical/final.jsonl \
  --metrics-out results/metrics/F5_binary_metrics.json
```

Regenerate figures after locally preparing the required result tables:

```bash
python -m eviosahs.plot_paper_figures \
  --results-dir results \
  --output-dir results/figures
```

## Data And Ethics

Raw clinical records and patient images are not distributed with this repository. Users must obtain data access through the appropriate institutional process and comply with all privacy, consent, and ethics requirements. Model weights must be downloaded separately under their upstream licenses.

## Citation

If you use this code, please cite the accompanying EviOSAHS manuscript. Update `CITATION.cff` with the final authors, venue, DOI, and repository URL after publication.

## License

The code is released under the MIT License unless replaced by an institution-approved license before public release. Data and model weights remain governed by their own access terms.

## Clinical Disclaimer

This repository is for research use only. It is not a medical device and must not be used for diagnosis, treatment, or clinical triage.
