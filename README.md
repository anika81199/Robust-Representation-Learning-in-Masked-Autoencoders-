# Robust Representation Learning in Masked Autoencoders

**Official RRPR Repository — ICPR 2026**

*Anika Shrivastava¹, Renu Rameshan², Samar Agnihotri¹*

¹ School of Computing and EE, IIT Mandi, HP – 175005, India
² Vehant Technologies Pvt. Ltd., UP – 201301, India

---

## Overview

This repository contains all source code, analysis scripts, and resources required to reproduce the experimental results presented in our ICPR 2026 paper **"Robust Representation Learning in Masked Autoencoders"**.

We study the internal representations learned by Masked Autoencoders (MAE), specifically the MAE ViT-Base architecture pretrained and fine-tuned on ImageNet-1K dataset. We first analyze the layer-wise token embeddings in the pretrained MAE encoder to examine how class-level structure evolves across network depth, with the aim of assessing whether class-relevant organization emerges in the absence of supervision. We then extend this analysis to the fine-tuned model and study the impact of input perturbations on both classification performance and the robustness of latent representations. Representation robustness is characterized using two complementary indicators: directional alignment between clean and perturbed embeddings, and feature-level robustness within individual attention heads.

We find that pretrained MAE progressively constructs a class-discriminative latent space without any supervision, that fine-tuned MAE maintains strong classification accuracy under heavy blur and attention-guided occlusion, and that two proposed sensitivity indicators — directional alignment (cosine similarity) and head-wise common-feature retention — closely track the breakdown in accuracy at extreme perturbation levels.

---

## Repository Highlights

- Layer-wise t-SNE visualization of CLS and mean patch token embeddings (Figure 1)
- Attention distance analysis comparing MAE vs. standard ViT (Figure 2)
- Subspace geometry analysis: principal angles and minimum singular values across layers (Figure 3)
- Classification accuracy under Gaussian blur with PSNR/SSIM reporting (Table 1)
- Script for creating Attention-guided occlusion
- Classification accuracy under attention-guided occlusion at 10 severity levels (Figure 4)
- Evaluation on ImageNet-A, ImageNet-R, and ImageNet-C distribution shifts
- Cosine similarity analysis between clean and perturbed embeddings (Table 2)
- Head-wise common-feature retention analysis under perturbations (Figure 5)
- Linear head fine-tuning on Caltech-256 using a frozen MAE encoder

---

## Repository Structure

```
RRPR_badge_MAE/
│
├── [Analysis scripts — this paper's contribution]
│   ├── tsne_visualizations.py          # Figure 1: t-SNE of CLS and patch tokens
│   ├── attn_maps.py                    # Figure 2: attention distance across layers
│   ├── subspace_geometry.py            # Figure 3: subspace geometry (principal angles, min SV)
│   ├── MAEClassifier.py                # Table 1 + Figure 4: blur & occlusion accuracy
│   ├── MAEClassifier_otherDatasets.py  # Section 4.2: accuracy on ImageNet-A/R/C
│   ├── attention_guided_occlusion.py   # Preprocessing: generates occluded image folders
│   ├── cosine_similarity_analysis.py   # Table 2: cosine similarity analysis
│   ├── common_features_analysis.py     # Figure 5: head-wise common feature retention
│   └── finetune_head_custom_dataset.py # Caltech-256 linear head fine-tuning
│
├── [MAE backbone — adapted from Meta AI's official MAE repo]
│   ├── models_mae.py                   # MAE model definition (modified to return attention)
│   ├── models_vit.py                   # ViT model definition
│   ├── engine_pretrain.py              # Pretraining engine
│   ├── main_finetune.py                # Full fine-tuning pipeline (requires timm==0.3.2)
│   ├── vit.py                          # ViT utilities
│   ├── _features.py                    # Feature extraction helpers (from timm)
│   └── util/                           # Learning rate, misc, dataset utilities
│
├── [Outputs]
│   └── final/                          # All generated figures are saved here
│
├── [Data — not included, see Dataset section]
│   ├── ImageNet/                       # 10-class ImageNet-1K subset
│   ├── Imagenet-C/                     # ImageNet-C corruptions
│   ├── imagenet-a/                     # ImageNet-A
│   ├── imagenet-r/                     # ImageNet-R
│   ├── 256_ObjectCategories/           # Caltech-256
│   └── occluded_{0,10,...,90}/         # Pre-generated attention-guided occluded images
│
├── mae_pretrain_vit_base.pth           # Pretrained MAE checkpoint (see Model Checkpoints)
├── mae_finetuned_vit_base.pth          # Fine-tuned MAE checkpoint (see Model Checkpoints)
├── imagenet1000_clsidx_to_labels.txt   # ImageNet class index to label mapping
├── requirements.txt
└── README.md
```

---

## What Is Our Contribution vs. the Original MAE Code

This repository is built on top of the [official MAE implementation by Meta AI](https://github.com/facebookresearch/mae). The backbone model files (`models_mae.py`, `models_vit.py`, `engine_pretrain.py`, `main_finetune.py`, and the `util/` directory) originate from that repository, with the following modifications:

- **`models_mae.py`** — Modified to return per-layer attention maps, queries, keys, values, and intermediate layer embeddings via a `return_attention=True` flag. These outputs are essential for the analysis pipelines in this paper.

All other `.py` files at the repository root are **original contributions** of this paper:

| File | Paper Section | What It Does |
|---|---|---|
| `tsne_visualizations.py` | §4.1 | Extracts layer-wise token embeddings and produces t-SNE plots |
| `attn_maps.py` | §4.1 | Computes and plots mean attention distances per head per layer |
| `subspace_geometry.py` | §4.1 | SVD-based subspace analysis; principal angles and min singular values |
| `attention_guided_occlusion.py` | §3.2, §4.2 | Attention rollout + patch occlusion; generates `occluded_*/` folders |
| `MAEClassifier.py` | §4.2 | Classification under clean, blurred, and occluded inputs |
| `MAEClassifier_otherDatasets.py` | §4.2 | Classification on ImageNet-A, ImageNet-R, ImageNet-C |
| `cosine_similarity_analysis.py` | §4.3 | Directional alignment between clean and perturbed embeddings |
| `common_features_analysis.py` | §4.3 | Head-wise common-feature retention under perturbations |
| `finetune_head_custom_dataset.py` | §4.2 | Freezes MAE encoder; trains only the linear classification head |

---

## Installation

### 1. Create a Conda Environment

```bash
conda create -n mae_rrpr python=3.10
conda activate mae_rrpr
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key library versions:**

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.5.1 |
| torchvision | 0.20.1 |
| timm | 1.0.22 |
| numpy | 1.23.5 |
| scipy | 1.14.1 |
| scikit-learn | 1.5.1 |
| scikit-image | 0.24.0 |
| seaborn | 0.13.2 |
| matplotlib | 3.9.2 |
| opencv-python | 4.10.0.84 |
| tqdm | 4.67.0 |
| pandas | 2.2.2 |

> **Note:** The file `main_finetune.py` is part of the original MAE repository and requires `timm==0.3.2`. It is **not** used by any of the paper's analysis scripts. All paper results use `finetune_head_custom_dataset.py` and the custom `MAEClassifier` class, which are compatible with `timm==1.0.22`.

---

## Hardware

All experiments were conducted on:

- **Machine:** Apple MacBook Air (M3)
- **CPU:** Apple M3
- **Memory:** 16 GB Unified Memory
- **Operating System:** macOS
- **PyTorch Backend:** CPU / Apple Metal Performance Shaders (MPS)

Most scripts automatically detect available hardware and use:
(`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## Model Checkpoints

Two checkpoints are required:

| Checkpoint | Description | Download |
|---|---|---|
| `mae_pretrain_vit_base.pth` | MAE ViT-Base pretrained on ImageNet-1K (75% masking) | [Meta AI MAE Release](https://github.com/facebookresearch/mae) — `ViT-Base` pretrained model |
| `mae_finetuned_vit_base.pth` | MAE ViT-Base fine-tuned for ImageNet classification | [Meta AI MAE Release](https://github.com/facebookresearch/mae) — `ViT-Base` fine-tuned model |

Download both `.pth` files and place them in the **repository root** (same level as `README.md`).

> The fine-tuned checkpoint was obtained from Meta AI's official MAE release and used directly for all robustness evaluations without further modification.

---

## Datasets

All datasets required to reproduce the paper's results are available via Google Drive:

> **[Download all data — Google Drive](https://drive.google.com/drive/folders/1DuAQra2Dh7xfTb9Xb-SSptn7cD1ecKah?usp=drive_link)**

The Google Drive folder contains the following, ready to use with no extra processing:

| Folder name in Drive | Contents | Used by |
|---|---|---|
| `ImageNet/` | 10-class ImageNet-1K subset (the fixed classes listed below) | All main experiments |
| `occluded_0/` … `occluded_90/` | 10 attention-guided occlusion folders, each containing the same 10 ImageNet classes at that occlusion level | `MAEClassifier.py` |
| `train/` | 200 images from the SAM dataset (used for attention distance validation) | `attn_maps.py` |
<!-- | `imagenet-a/` | ImageNet-A adversarial natural images | `MAEClassifier_otherDatasets.py` |
| `imagenet-r/` | ImageNet-R artistic renditions | `MAEClassifier_otherDatasets.py` |
| `Imagenet-C/` | ImageNet-C corruptions (15 types × 5 severities) | `MAEClassifier_otherDatasets.py` | -->

Download and extract all folders into the **repository root**, preserving the folder names exactly as listed above.

---

### ImageNet-1K (10-class subset)

The 10 fixed classes used across all ImageNet experiments are:

| Class ID | Synset | Description |
|---|---|---|
| 235 | n02106662 | German shepherd |
| 98 | n01855032 | Red-breasted merganser |
| 237 | n02107312 | Miniature pinscher |
| 229 | n02105641 | Old English sheepdog |
| 417 | n02782093 | Balloon |
| 421 | n02788148 | Bannister |
| 430 | n02802426 | Basketball |
| 668 | n03788195 | Mosque |
| 757 | n04065272 | Recreational vehicle |
| 814 | n04273569 | Speedboat |

Expected directory structure after extracting from Google Drive:

```
ImageNet/
├── n02106662/   ← class 235
├── n01855032/   ← class 98
├── n02107312/   ← class 237
├── n02105641/   ← class 229
├── n02782093/   ← class 417
├── n02788148/   ← class 421
├── n02802426/   ← class 430
├── n03788195/   ← class 668
├── n04065272/   ← class 757
└── n04273569/   ← class 814
```

### Attention-Guided Occluded Images

The 10 `occluded_*/` folders are pre-generated using attention rollout and are included in the Google Drive. Each folder corresponds to one occlusion level (0%, 10%, ..., 90%) and contains one subfolder per class:

```
occluded_50/
├── n02106662/   ← 50%-occluded images of class 235
├── n01855032/
...
```

If you prefer to regenerate them from scratch, uncomment the generation loop in `attention_guided_occlusion.py` (lines 256–297) and run:

```bash
python attention_guided_occlusion.py
```

### SAM Images (`train/`)

The `train/` folder contains 200 images from the [Segment Anything (SAM) dataset](https://segment-anything.com/dataset/index.html), used in `attn_maps.py` to validate that MAE's global attention pattern generalises beyond ImageNet.

### ImageNet-A, ImageNet-R, ImageNet-C

These are available in the Google Drive. They can also be downloaded from their original sources:

- ImageNet-C: [github.com/hendrycks/robustness](https://github.com/hendrycks/robustness) → place at `Imagenet-C/`
- ImageNet-A / ImageNet-R: [github.com/hendrycks/imagenet-r](https://github.com/hendrycks/imagenet-r) → place at `imagenet-a/` and `imagenet-r/`

### Caltech-256 (optional — for head fine-tuning only)

Download from [Caltech 256](https://data.caltech.edu/records/nyy15-4j048). Place at `256_ObjectCategories/`. Only required to run `finetune_head_custom_dataset.py`.

---

## Reproducing Results

Create the output directory before running any script:

```bash
mkdir -p final
```

All scripts are run from the **repository root**.

---

### Figure 1 — t-SNE Visualization of Token Embeddings (Section 4.1)

```bash
python tsne_visualizations.py
```

**Outputs:**
- `final/tsne_class_tokens.png` — Figure 1(a): CLS token t-SNE across all 12 layers
- `final/tsne_mean_patch_tokens.png` — Figure 1(b): Mean patch token t-SNE across all 12 layers

---

### Figure 2 — Attention Distance Analysis (Section 4.1)

```bash
python attn_maps.py
```

**Outputs:**
- `final/mae_attn_dist_mean_across_imgs.png` — Figure 2 (right): MAE mean attention distance per head per layer
- `final/sam_attention_boxplots.png` — SAM attention distance (supplementary)

---

### Figure 3 — Subspace Geometry (Section 4.1)

```bash
python subspace_geometry.py
```

**Outputs:**
- `final/theta1_boxplot.png` — Figure 3 (left): Layer-wise distribution of principal angles θ₁
- `final/minSVplot.png` — Figure 3 (right): Layer-wise evolution of minimum singular value

---

### Table 1 + Figure 4 — Classification Under Blur and Occlusion (Section 4.2)

> Requires the `occluded_*/` folders (see Preprocessing above).

```bash
python MAEClassifier.py
```

**Outputs:**
- `final/occ_acc.png` — Figure 4: Occlusion level vs. mean accuracy
- Printed table: Top-1 accuracy, PSNR, and SSIM for each Gaussian blur setting (Table 1)
- Printed per-class and mean accuracy for each occlusion level

---

### Section 4.2 — Classification on ImageNet-A, ImageNet-R, ImageNet-C

```bash
python MAEClassifier_otherDatasets.py
```

**Output:** Printed mean accuracy for each dataset/corruption type/severity level.

---

### Table 2 — Cosine Similarity Analysis (Section 4.3)

```bash
python cosine_similarity_analysis.py
```

**Outputs:**
- CSV file with per-image cosine similarities
- Printed mean cosine similarity for each perturbation level (Table 2a: blur, Table 2b: occlusion)

---

### Figure 5 — Common Feature Retention Analysis (Section 4.3)

```bash
python common_features_analysis.py
```

**Outputs:**
- Feature retention heatmaps across layers and heads for clean, blurred, and occluded inputs
- Average drop plots — Figure 5(a): blur, Figure 5(b): occlusion

---

### Caltech-256 Linear Head Fine-Tuning (Section 4.2)

```bash
python finetune_head_custom_dataset.py
```

Uses default settings: `train_per_class=60`, `val_per_class=10`, `test_per_class=10`, `epochs=20`, `lr=1e-3`, `batch_size=32`. Custom options:

```bash
python finetune_head_custom_dataset.py \
    --data_dir 256_ObjectCategories \
    --checkpoint mae_finetuned_vit_base.pth \
    --train_per_class 60 --val_per_class 10 --test_per_class 10 \
    --epochs 20 --lr 1e-3 --batch_size 32
```

**Output:** Printed top-1 accuracy on the Caltech-256 test split.

---

## Figure Reproduction Summary

| Paper Element | Script | Output File(s) |
|---|---|---|
| Figure 1(a) — CLS t-SNE | `tsne_visualizations.py` | `final/tsne_class_tokens.png` |
| Figure 1(b) — Patch t-SNE | `tsne_visualizations.py` | `final/tsne_mean_patch_tokens.png` |
| Figure 2 — Attention distance | `attn_maps.py` | `final/mae_attn_dist_mean_across_imgs.png` |
| Figure 3 (left) — Principal angles | `subspace_geometry.py` | `final/theta1_boxplot.png` |
| Figure 3 (right) — Min singular value | `subspace_geometry.py` | `final/minSVplot.png` |
| Table 1 — Blur accuracy + PSNR/SSIM | `MAEClassifier.py` | Printed output |
| Figure 4 — Occlusion accuracy curve | `MAEClassifier.py` | `final/occ_acc.png` |
| Table 2(a) — Cosine sim. under blur | `cosine_similarity_analysis.py` | Printed + CSV |
| Table 2(b) — Cosine sim. under occlusion | `cosine_similarity_analysis.py` | Printed + CSV |
| Figure 5(a) — Feature drop under blur | `common_features_analysis.py` | `final/blurs_drops.png` |
| Figure 5(b) — Feature drop under occlusion | `common_features_analysis.py` | `final/occ_drops.png`|

---

## Runtime Estimates

| Script | Approximate Runtime (GPU) |
|---|---|
| `tsne_visualizations.py` | ~5 min |
| `attn_maps.py` | ~3 min |
| `subspace_geometry.py` | ~5 min |
| `attention_guided_occlusion.py` (generation) | ~20–30 min |
| `MAEClassifier.py` | ~10 min |
| `MAEClassifier_otherDatasets.py` | ~10–15 min |
| `cosine_similarity_analysis.py` | ~5 min |
| `common_features_analysis.py` | ~5 min |
| `finetune_head_custom_dataset.py` | ~300 min (20 epochs) |

---

## Reproducibility Statement

This repository contains:

- Source code for all paper analyses
- Pretrained and fine-tuned model checkpoints (via Meta AI's official MAE release)
- Complete dependency list with pinned versions (`requirements.txt`)
- Analysis scripts with inline documentation linking each script to its paper section
- Preprocessing script for generating occluded images
- All figure reproduction scripts producing outputs that match the paper
- Dataset preparation instructions 

---

## Acknowledgements

This repository builds upon the official Masked Autoencoder (MAE) implementation released by Meta AI Research:

> He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). Masked Autoencoders Are Scalable Vision Learners. *arXiv:2111.06377*.
> Official repository: https://github.com/facebookresearch/mae

The model files (`models_mae.py`, `models_vit.py`, `engine_pretrain.py`, `main_finetune.py`, `util/`) originate from that repository. `models_mae.py` was modified to expose per-layer attention maps and intermediate embeddings. All analysis scripts, visualization tools, robustness evaluation pipelines, and the `finetune_head_custom_dataset.py` script were developed by Anika Shrivastava as original contributions of this paper.

---

## License

This repository retains the original MAE license:
**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

See [LICENSE](LICENSE) for the full license text.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{shrivastava2026robust,
  title     = {Robust Representation Learning in Masked Autoencoders},
  author    = {Shrivastava, Anika and Rameshan, Renu and Agnihotri, Samar},
  booktitle = {Proceedings of ICPR},
  year      = {August, 2026}
}
```
