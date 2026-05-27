# SegMoTE

Official implementation staging directory for **SegMoTE**, a mixture-of-experts
model for interactive medical image segmentation.

This release directory contains the training and evaluation code derived from
the research implementation. Large training data, intermediate visualizations,
private filesystem paths, duplicate baseline copies, and experimental analysis
scripts are deliberately excluded.


## Environment

The original experiments used Python, PyTorch, TIMM, Transformers, MONAI,
OpenCV and common scientific Python packages. Install the Python dependencies
with:

```bash
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation appropriate for your machine.

## Data

Each dataset is expected under `dataset/<DATASET_NAME>/` with a
`dataset.json` file and the image/annotation paths referenced by that file.
For multiple datasets, provide their names after `--dataset_list`.

```text
dataset/
`-- BTCV/
    |-- dataset.json
    |-- image/
    |-- label/
    `-- imask/
```

## Training

Training can start from a user-supplied initialization checkpoint:

```bash
python train.py \
  --data_dir dataset \
  --dataset_list BTCV \
  --sam_checkpoint /path/to/initialization.pth \
  --task_name segmote_train
```

For distributed execution, additionally pass `--dist --multi_gpu` and the
desired `--gpu_ids`.


## Citation And License

```bibtex
@article{lu2026segmote,
  title={SegMoTE: Token-Level Mixture of Experts for Medical Image Segmentation},
  author={Lu, Yujie and Li, Jingwen and Ju, Sibo and Su, Yanzhou and Liu, Yisong and Zhu, Min and Cheng, Junlong and others},
  journal={arXiv preprint arXiv:2602.19213},
  year={2026}
}
