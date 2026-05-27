<div align="center">

# SegMoTE: Token-Level Mixture of Experts for Medical Image Segmentation

**Yujie Lu<sup>1*</sup>, Jingwen Li<sup>2*</sup>, Sibo Ju<sup>3</sup>, Yanzhou Su<sup>4</sup>,  
He Yao<sup>1</sup>, Yisong Liu<sup>1</sup>, Min Zhu<sup>1&dagger;</sup>, Junlong Cheng<sup>1&dagger;</sup>**

<sup>1</sup>Sichuan University &nbsp;&nbsp;
<sup>2</sup>Xinjiang University &nbsp;&nbsp;
<sup>3</sup>Fuzhou University &nbsp;&nbsp;
<sup>4</sup>Alibaba DAMO Academy

<sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup> Corresponding authors

**CVPR 2026**

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b.svg)](#citation)
[![Code](https://img.shields.io/badge/Code-PyTorch-blue.svg)](#installation)
[![Model](https://img.shields.io/badge/Model-Coming%20Soon-orange.svg)](#checkpoint)

</div>

## Abstract

Medical image segmentation requires robust adaptation across heterogeneous
modalities and anatomical structures, while pixel-level annotation remains
expensive. SegMoTE is an efficient adaptation framework built on the Segment
Anything Model (SAM). It introduces a **token-level Mixture of Experts
(MoTE)** mechanism that dynamically selects modality-adaptive expert tokens,
and a **Progressive Prompt Tokenization (PPT)** module that learns
feature-conditioned prompts for prompt-free segmentation on suitable
foreground-background tasks. Trained on the curated **MedSeg-HQ** dataset,
SegMoTE aims to retain the flexible prompt interface and generalization ability
of SAM while providing lightweight adaptation for multimodal medical image
segmentation.

<p align="center">
  <img src="figures/Introduction.png" width="98%" alt="SegMoTE motivation and comparison with previous SAM adaptation methods">
</p>

## Architecture

<p align="center">
  <img src="figures/SegMoTE.png" width="100%" alt="Overall architecture of SegMoTE">
</p>

SegMoTE extends SAM with two components. First, **MoTE** injects learnable
expert tokens into the mask decoder and uses token-level routing to select
specialized experts for different modalities and tasks. A load-balancing
objective encourages effective expert utilization. Second, **PPT** pools image
features into adaptive prompt tokens for selected few-class segmentation
settings, reducing dependence on manual prompts. The framework supports point,
bounding-box, and text prompts while retaining an efficient inference path.

## Highlights

- **Token-level expert routing:** SegMoTE dynamically activates expert tokens
  for modality- and task-adaptive segmentation.
- **Progressive prompt tokenization:** Feature-conditioned prompt tokens
  support automatic segmentation for suitable binary foreground-background
  tasks.
- **Multimodal medical segmentation:** The framework is designed for medical
  datasets spanning CT, MRI, dermoscopy, X-ray, and other modalities.
- **SAM-compatible interaction:** Point, bounding-box, and text prompts are
  supported in the released implementation.

## Updates

- **May 2026:** Code release preparation and inference checkpoint packaging.
- **May 2026:** Deterministic MoTE routing enabled during evaluation.

## Installation

Create an environment with a CUDA-enabled PyTorch installation appropriate for
your hardware, then install the remaining dependencies:

```bash
git clone <repository-url>
cd SegMoTE
pip install -r requirements.txt
```

The implementation uses PyTorch, TIMM, Transformers, MONAI, OpenCV, and common
scientific Python packages.

## Data Preparation

The paper uses **MedSeg-HQ**, a curated multimodal medical segmentation
collection. Dataset files are not included in this repository.

Each dataset used by the loader should be organized as follows:

```text
dataset/
`-- <DATASET_NAME>/
    |-- dataset.json
    |-- image/
    |-- label/
    `-- imask/
```

The `dataset.json` file provides class definitions and references to the
training and testing samples. Multiple datasets can be supplied with
`--dataset_list`.

## Checkpoint

Two checkpoint files are required for evaluation:

```text
checkpoints/IMISNet-B.pth
checkpoints/segmote_best.pth
```

| Checkpoint | Usage |
| --- | --- |
| `IMISNet-B.pth` | Base initialization checkpoint loaded with `--sam_checkpoint` before loading SegMoTE weights. |
| `segmote_best.pth` | SegMoTE inference checkpoint loaded with `--pretrain_path`. |

Download the checkpoints from Baidu Netdisk:

```text
Link: <BAIDU_NETDISK_DOWNLOAD_LINK>
Extraction code: <BAIDU_NETDISK_EXTRACTION_CODE>
```

After downloading, place both checkpoint files in the `checkpoints/`
directory. The SegMoTE inference artifact contains only `model_state_dict`;
training metadata such as epoch counters and optimizer state has been removed.
As a result, evaluation loads the model without displaying stored training
epochs.

The checkpoint files are large and are intentionally excluded from the source
repository.

Legacy research checkpoints can also be converted to the public parameter
naming scheme:

```bash
python tools/release_checkpoint.py /path/to/legacy.pth checkpoints/segmote_best.pth
```

## Evaluation

Evaluate the released checkpoint on a dataset with bounding-box prompts:

```bash
python test.py \
  --data_dir dataset \
  --dataset_list BTCV \
  --sam_checkpoint checkpoints/IMISNet-B.pth \
  --pretrain_path checkpoints/segmote_best.pth \
  --prompt_mode bboxes \
  --output_dir outputs/BTCV
```

Other supported prompt modes are `points` and `text`:

```bash
python test.py \
  --data_dir dataset \
  --dataset_list BTCV \
  --sam_checkpoint checkpoints/IMISNet-B.pth \
  --pretrain_path checkpoints/segmote_best.pth \
  --prompt_mode points \
  --output_dir outputs/BTCV_points
```

## Training

Train SegMoTE from the base initialization checkpoint:

```bash
python train.py \
  --data_dir dataset \
  --dataset_list BTCV \
  --sam_checkpoint checkpoints/IMISNet-B.pth \
  --task_name segmote_train
```

For distributed training on multiple GPUs:

```bash
python train.py \
  --data_dir dataset \
  --dataset_list BTCV \
  --sam_checkpoint checkpoints/IMISNet-B.pth \
  --task_name segmote_train \
  --dist \
  --multi_gpu \
  --gpu_ids 0 1 2 3 4 5 6 7
```

## Repository Structure

```text
SegMoTE/
|-- checkpoints/
|-- dataloaders/
|-- figures/
|-- segment_anything/
|-- tools/
|   `-- release_checkpoint.py
|-- checkpoint_utils.py
|-- data_loader.py
|-- model.py
|-- test.py
|-- train.py
|-- requirements.txt
`-- utils.py
```

## Citation

The BibTeX entry will be updated after the public paper record is available:

```bibtex
@inproceedings{lu2026segmote,
  title     = {SegMoTE: Token-Level Mixture of Experts for Medical Image Segmentation},
  author    = {Lu, Yujie and Li, Jingwen and Ju, Sibo and Su, Yanzhou and Yao, He and Liu, Yisong and Zhu, Min and Cheng, Junlong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

## Acknowledgements

This repository contains components derived from the Segment Anything Model
(SAM). Please follow the applicable upstream license terms when distributing
or modifying those components.
