# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional

from .common import LayerNorm2d
class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            semantic_out_dim: int = 512,
    ) -> None:


        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_expert_tokens = 4
        self.vit_dim = 768
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.semantic_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.expert_tokens = nn.Embedding(self.num_expert_tokens, transformer_dim)
        self.text_token = nn.Embedding(1, transformer_dim)
        self.boxApoint_token = nn.Embedding(1, transformer_dim)
        self.expert_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(self.vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.expert_embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.expert_feature_fusion = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.expert_output_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_expert_tokens)
            ]
        )

        self.text_output = nn.Linear(transformer_dim, transformer_dim // 8)
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens,
                                       iou_head_depth)
        self.semantic_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, semantic_out_dim,
                                            iou_head_depth)
        self.img_prompt_pool = AttnPoolTokens(channels=768, token_size=2, num_heads=8, use_posenc=True)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            text_prompt_embeddings: Optional[torch.Tensor],
            multimask_output: bool,
            interm_embeddings: torch.Tensor,
            inter:bool,
            if_prompt,
            test_mode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        masks, iou_pred, semantic_pred,loss,moe_idx = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            text_prompt_embeddings=text_prompt_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            interm_embeddings=interm_embeddings,
            inter=inter,
            if_prompt= if_prompt,
            test_mode=test_mode
        )


        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks_selected = masks[torch.arange(masks.size(0)), moe_idx, :, :]
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        semantic_pred = semantic_pred[:, mask_slice, :]
        outputs = {'low_res_masks': masks_selected, 'iou_pred': iou_pred, 'semantic_pred': semantic_pred,
                   'candidate_masks': masks}
        return outputs,loss,moe_idx

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            text_prompt_embeddings: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            interm_embeddings: torch.Tensor,
            inter: bool,
            if_prompt,
            test_mode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight, self.semantic_tokens.weight, self.expert_tokens.weight],
            dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        random_prob = torch.rand(1).item()
        if if_prompt:
            sparse_prompt_embeddings = self.img_prompt_pool(image_embeddings)


        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        num_tokens = tokens.shape[1]
        sparse_dim = sparse_prompt_embeddings.shape[1]

        vit_features = interm_embeddings.permute(0, 3, 1,
                                                 2)
        expert_features = self.expert_embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)


        src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0],
                                          dim=0)
        b, c, h, w = src.shape


        hs, src,loss,moe_idx = self.transformer(src, pos_src, tokens,inter,test_mode)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]
        semantic_tokens_out = hs[:, (1 + self.num_mask_tokens):(num_tokens - sparse_dim - self.num_expert_tokens), :]
        expert_tokens_out = hs[:, (num_tokens - sparse_dim - self.num_expert_tokens):(num_tokens - sparse_dim), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        expert_embedding = self.expert_feature_fusion(upscaled_embedding) + expert_features

        hyper_in_list: List[torch.Tensor] = []
        expert_hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        for i in range(self.num_expert_tokens):
            expert_hyper_in_list.append(self.expert_output_mlps[i](expert_tokens_out[:, i, :]))
        expert_hyper_in = torch.stack(expert_hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        expert_masks = (expert_hyper_in @ expert_embedding.view(b, c, h * w)).view(b, -1, h, w)
        if text_prompt_embeddings is not None:
            upscaled_embedding = expert_embedding.view(b, c, h * w)
            text_prompt_embeddings = self.text_output(text_prompt_embeddings.unsqueeze(1))
            text_mask = (text_prompt_embeddings @ upscaled_embedding).view(b, -1, h, w)
            text_mask = text_mask.repeat(1, expert_masks.shape[1], 1, 1)
            expert_masks = expert_masks + text_mask


        iou_pred = self.iou_prediction_head(iou_token_out)
        semantic_pred = self.semantic_prediction_head(semantic_tokens_out)

        return expert_masks, iou_pred, semantic_pred,loss,moe_idx

class AttnPoolTokens(nn.Module):


    def __init__(self, channels=768, token_size=4, num_heads=8, use_posenc=True):
        super().__init__()
        self.token_size = token_size
        self.query = nn.Parameter(torch.randn(token_size, channels))
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ln_in  = nn.LayerNorm(channels)
        self.ln_out = nn.LayerNorm(channels)
        self.use_posenc = use_posenc
        if use_posenc:
            self.pos_proj = nn.Linear(2, channels)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels*4),
            nn.GELU(),
            nn.Linear(channels*4, channels),
        )

    def _build_2d_pos(self, H, W, device):
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        coord = torch.stack([xs, ys], dim=-1).view(H*W, 2)
        return self.pos_proj(coord)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = x.flatten(2).transpose(1, 2)
        feats = self.ln_in(feats)

        if self.use_posenc:
            pos = self._build_2d_pos(H, W, x.device)
            feats = feats + pos.unsqueeze(0)

        Q = self.query.unsqueeze(0).expand(B, -1, -1)
        tokens, _ = self.attn(Q, feats, feats)
        tokens = self.ln_out(tokens + self.mlp(tokens))
        return tokens


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        for i, layer in enumerate(self.layers):


            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
