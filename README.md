# SegMoTE

Official implementation staging directory for SegMoTE: Token-Level Mixture of Experts for Medical Image Segmentation

## Environment

The original experiments used Python, PyTorch, TIMM, Transformers, MONAI,
OpenCV and common scientific Python packages. Install the Python dependencies
with:

```bash
pip install -r requirements.txt
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


## Citation And License

```bibtex
@article{lu2026segmote,
  title={SegMoTE: Token-Level Mixture of Experts for Medical Image Segmentation},
  author={Lu, Yujie and Li, Jingwen and Ju, Sibo and Su, Yanzhou and Liu, Yisong and Zhu, Min and Cheng, Junlong and others},
  journal={arXiv preprint arXiv:2602.19213},
  year={2026}
}
