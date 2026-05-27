# SegMoTE

Official implementation staging directory for **SegMoTE**, a mixture-of-experts
model for interactive medical image segmentation.

This release directory contains the training and evaluation code derived from
the research implementation. Large training data, intermediate visualizations,
private filesystem paths, duplicate baseline copies, and experimental analysis
scripts are deliberately excluded.

## Repository Layout

```text
SegMoTE/
|-- checkpoints/                 # Put downloaded weights here (ignored by Git)
|-- dataloaders/
|-- segment_anything/            # SegMoTE encoder/prompt/decoder implementation
|-- tools/release_checkpoint.py  # Legacy checkpoint name converter
|-- checkpoint_utils.py          # Runtime compatibility for legacy weights
|-- data_loader.py
|-- model.py
|-- test.py
|-- train.py
`-- utils.py
```

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

## Naming Compatibility

The research workspace originated from an earlier interactive segmentation
baseline. The released model API uses `SegMoTE` and `SegMoTEPredictor`, and
the decoder's learnable MoE output branch uses `expert_*` identifiers. Legacy
names occur only in the compatibility map required to load previously trained
weights.

## Citation And License

Add the final SegMoTE paper citation, checkpoint download URL, dataset access
instructions, and the intended open-source license before publishing this
directory publicly. The `segment_anything`-derived components retain their
upstream copyright headers and must be distributed consistently with their
applicable license terms.
