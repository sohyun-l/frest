# FREST: Feature RESToration for Semantic Segmentation under Multiple Adverse Conditions

**[ECCV 2024] Official PyTorch implementation of FREST.**

[![arXiv](https://img.shields.io/badge/arXiv-2407.13437-b31b1b.svg)](https://arxiv.org/abs/2407.13437)
[![Project Page](https://img.shields.io/badge/Project-Page-1f72b1.svg)](https://sohyun-l.github.io/frest)
[![Conference](https://img.shields.io/badge/ECCV-2024-6f42c1.svg)](https://eccv.ecva.net/Conferences/2024)

FREST is a source-free domain adaptation (SFDA) framework for semantic
segmentation under adverse conditions. It alternates two steps: (1) learning a
**condition embedding space** that isolates condition-specific information, and
(2) **restoring** features of adverse-condition images toward the normal
condition on that space. This repository covers the main **Cityscapes → ACDC**
setting.

## Method overview

| Paper component | Where in the code |
| --- | --- |
| Condition strainer `psi_strainer` (Fig. 3) | `models/backbones/strainer.py` (`ConditionStrainer`), attached per stage in `models/backbones/mix_transformer_strainer.py` |
| Projection onto the condition embedding space `psi_proj` | `models/heads/projection.py` (`ProjectionHead`) + `models/backbones/condition_projector.py` (`ConditionProjector`) |
| Condition-specific loss `L_spec` (Eq. 1–2) | sampling in `models/losses.py` (`ConditionContrastiveSampler`), contrastive term in `FREST.training_step` (Step 1) |
| Feature restoration loss `L_resto` (Eq. 3) | `restoration_loss` in `FREST.training_step` (Step 2) |
| Adverse-condition discriminating loss `L_dis` (Eq. 4) | `discriminate=True` path in the backbone; `discriminating_loss` in `FREST.training_step` |
| Self-training `L_self` (CBST) + entropy `L_ent` | `helpers/pseudo_labels.py`, `models/losses.py` (`NormalizedEntropyLoss`) |
| Restored features used at inference (encoder + decoder only) | `inference=True` path in the backbone (the strainer is skipped) |

The two steps are alternated every iteration; only the encoder and decoder are
used for evaluation.

## Setup

```bash
conda create -n frest python=3.8 -y && conda activate frest
pip install -r requirements.txt
```

A CUDA toolchain is required: the optical-flow correlation op
(`models/correlation_ops`) is JIT-compiled on first use (needs `nvcc` + `ninja`).

## Data and weights

Set `DATA_DIR` to a folder laid out as:

```
$DATA_DIR/
  ACDC/                 # rgb_anon/, gt/  (https://acdc.vision.ee.ethz.ch/)
  pseudo_labels/
    pseudo_labels_train_ACDC_cma_segformer/   # auto-downloaded on first run
```

- **Source model**: place the Cityscapes-pretrained SegFormer-B5 at
  `./pretrained_models/segformer.b5.1024x1024.city.160k.pth`
  (official SegFormer weights). [download link](https://drive.google.com/file/d/1UlBXN5LLLKLsj51kq8DxHt5lzXw4x96k/view?usp=sharing)
- **Alignment** (training only): place the UAWarpC MegaDepth weights at
  `./pretrained_models/uawarpc_megadepth.ckpt`. Not needed for evaluation. [download link](https://drive.google.com/file/d/1qMngnVPUroFOYpBvwlxNCWpjtr3bNBfD/view?usp=sharing)

See `pretrained_models/README.md` for download links of the two files above.

## Train

```bash
DATA_DIR=/path/to/data bash scripts/train_acdc.sh
```

## Evaluate

```bash
DATA_DIR=/path/to/data bash scripts/eval_acdc.sh checkpoints/frest_acdc.ckpt
```

## FREST checkpoint

Download the FREST checkpoint and place it at `checkpoints/frest_acdc.ckpt`:

- **Checkpoint (Cityscapes → ACDC):** [download link](https://drive.google.com/file/d/16NDm5lNqGpg7kbkGYJ8eBGs57NK_luvx/view?usp=sharing)

If you instead have a checkpoint from the original training code (legacy module
names), convert its keys to this release's layout with:

```bash
python convert_checkpoint.py SRC.ckpt checkpoints/frest_acdc.ckpt
```

## Citation

```bibtex
@inproceedings{lee2024frest,
  title     = {FREST: Feature RESToration for Semantic Segmentation under Multiple Adverse Conditions},
  author    = {Lee, Sohyun and Kim, Namyup and Kim, Sungyeon and Kwak, Suha},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```

## Acknowledgements

The codebase builds on [Refign/CMA](https://github.com/brdav/cma) and
[SegFormer](https://github.com/NVlabs/SegFormer).
