import torch
from functools import partial
from .modeling import ViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
from torch.nn import functional as F
from checkpoint_utils import upgrade_legacy_state_dict

def build_sam_vit_h(args):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        pretrain_model = 'samvit_huge_patch16'
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        pretrain_model = 'samvit_large_patch16'
    )


def build_sam_vit_b(args):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        pretrain_model = 'samvit_base_patch16'

    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

def _normalize_ckpt_keys(sd_keys):

    norm = set()
    for k in sd_keys:
        if k.startswith("module."): k = k[len("module."):]
        if k.startswith("sam."):    k = k[len("sam."):]
        if k.startswith("model."):  k = k[len("model."):]
        norm.add(k)
    return norm

def _present_in_ckpt(name: str, ckpt_keys: set) -> bool:

    if name in ckpt_keys:
        return True
    for k in ckpt_keys:
        if k.endswith(name):
            return True
    return False

def _freeze_original_keep_new_trainable(sam, ckpt_state_dict, only_mask_decoder=True, freeze_encoder=True):


    ckpt_keys = _normalize_ckpt_keys(ckpt_state_dict.keys())


    if freeze_encoder:
        for p in sam.image_encoder.parameters():
            p.requires_grad = False


    freeze_cnt = train_cnt = 0
    for name, p in sam.named_parameters():

        if name.startswith("image_encoder."):
            continue

        if only_mask_decoder and not name.startswith("mask_decoder."):
            continue

        if _present_in_ckpt(name, ckpt_keys):

            p.requires_grad = False
            freeze_cnt += p.numel()
        else:

            p.requires_grad = True
            train_cnt += p.numel()


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint,
    pretrain_model
):
    prompt_embed_dim = 768
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder = ViT(
            encoder_embed_dim = encoder_embed_dim,
            pretrain_model= pretrain_model,
            out_chans= prompt_embed_dim,
            depth = encoder_depth,
            freeze_encoder = True,
            pretrained=False,
            ),

        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),

        text_model =  CLIPTextModel(CLIPTextConfig()),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None:
        state_dict = torch.load(open(checkpoint, "rb"), map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        state_dict = upgrade_legacy_state_dict(state_dict)
        sam.load_state_dict(state_dict, strict=False)
        print(f"Loaded initialization checkpoint from {checkpoint}")


        _freeze_original_keep_new_trainable(
            sam,
            ckpt_state_dict=state_dict,
            only_mask_decoder=True,
            freeze_encoder=True
        )

    return sam
