# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import Tensor, nn
import numpy as np
import math
from typing import Tuple, Type
from torch.distributions.normal import Normal
from .common import MLPBlock,MLPBlock_MoE,MLPBlock_token
from collections import defaultdict
import random
import pandas as pd
class SparseDispatcher_token(object):


    def __init__(self, num_experts, gates):


        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        _, self._expert_index = sorted_experts.split(1, dim=1)

        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]

        self._part_sizes = (gates > 0).sum(0).tolist()

        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):


        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, top_logits, multiply_by_gates=True):


        stitched = torch.cat(expert_out, 0)

        gates = self._gates[self._expert_index]
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)

        combined = zeros.index_add(0, self._batch_index, stitched.float())

        combined[combined == 0] = np.finfo(float).eps

        return combined

    def expert_to_gates(self):


        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,

    ) -> None:


        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        inter: bool,
        test_mode,
    ) -> Tuple[Tensor, Tensor]:


        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)


        queries = point_embedding
        keys = image_embedding


        for layer in self.layers:
            queries, keys,loss,moe_idx = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                inter=inter,
                test_mode=test_mode
            )


        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys,loss,moe_idx


class TwoWayAttentionBlock(nn.Module):
    def     __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        last_cross_attn_attn=None,
        last_gates_token = None,
        last_hw = None,
        last_t_slice = None,
    ) -> None:


        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.noisy_gating=True
        self.num_experts = 4


        self.w_gate_token = nn.Parameter(torch.zeros(768, self.num_experts), requires_grad=True)
        self.w_noise_token = nn.Parameter(torch.zeros(768, self.num_experts), requires_grad=True)

        self.last_gates_token = last_gates_token
        self.last_hw = last_hw
        self.last_t_slice = last_t_slice
        self.last_cross_attn_attn = last_cross_attn_attn

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)


        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        rank = 64
        emb = 768

        self.skip_first_layer_pe = skip_first_layer_pe
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.k=1

        self.experts_token = nn.ModuleList([MLPBlock_token() for i in range(self.num_experts)])
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.dte = nn.Linear(rank, rank)
        self.dteall = nn.Linear(rank, rank)


        self.dte_token = nn.Linear(rank, rank)
        self.dteall_token = nn.Linear(rank, rank)
        self.proj_token = nn.Linear(rank, emb)


    def cv_squared(self, x):


        eps = 1e-10


        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):


        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):


        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating_token(self, x, test_mode, noise_epsilon=1e-2):

        B, T, D = x.shape
        gates_list = []
        top_k_logits_list = []
        softmax_logits_list = []
        top_k_indices_list = []
    
        for i in range(B):
            x_i = x[i]
    
            clean_logits = x_i @ self.w_gate_token
    
            if self.noisy_gating and test_mode == False:
                raw_noise_stddev = x_i @ self.w_noise_token
                noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
                noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
                logits = noisy_logits
            else:
                logits = clean_logits
    
            top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
            top_k_logits = top_logits[:, :self.k]
            top_k_indices = top_indices[:, :self.k]
    
            probs = self.softmax(logits)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
    
            zeros = torch.zeros_like(logits)
            hard_gates = zeros.scatter(1, top_k_indices, 1.0)
            soft_gates = zeros.scatter(1, top_k_indices, top_k_gates)
            gates = hard_gates - soft_gates.detach() + soft_gates
    
            gates_list.append(gates)
            top_k_logits_list.append(top_k_logits)
            softmax_logits_list.append(probs)
            top_k_indices_list.append(top_k_indices)
    
        gates = torch.stack(gates_list, dim=0)
        top_k_logits = torch.stack(top_k_logits_list, dim=0)
        logits = torch.stack(softmax_logits_list, dim=0)
        top_k_indices = torch.stack(top_k_indices_list, dim=0)
    
        if self.noisy_gating and self.k < self.num_experts and test_mode == False:
            load = gates.sum(dim=(0, 1))
        else:
            load = self._gates_to_load(gates.view(-1, self.num_experts))
    
        return gates, load, logits, top_k_logits, top_k_indices


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor,test_mode,inter=False,loss_coef=1e-1
    ) -> Tuple[Tensor, Tensor]:

        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        self.cross_attn_token_to_image.save_attn = True

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        self.cross_attn_token_to_image.save_attn = False
        self.last_cross_attn_attn = self.cross_attn_token_to_image.attn
        queries = queries + attn_out
        queries = self.norm2(queries)
        idx_moe = self.num_experts +9
        queries_moe =queries[:, :idx_moe, :]
        sp_token = queries[:, idx_moe:, :]
        queries_front = queries_moe[:, :9, :]
        queries_back = queries_moe[:, -self.num_experts:, :]


        self.last_t_slice = slice(9, 13)

        queries_sam = torch.cat([queries_front, sp_token], dim=1)
        mlp_out = self.mlp(queries_sam)
        queries_sam = queries_sam + mlp_out
        queries_sam = self.norm3(queries_sam)

        loss_token=None
        max_token_idx=None
        if inter==False:
            gates_token, load_token, logits_token, top_k_logits_token, top_k_indexs_token = self.noisy_top_k_gating_token(
                queries_back, test_mode,)
            self.last_gates_token = gates_token.detach()
            self.last_hw = (32, 32)


            token_scores = top_k_logits_token.squeeze(-1)
            max_scores, max_token_idx = torch.max(token_scores, dim=1)
            if self.k >1:
                tok_max_scores, tok_argmax_k = torch.max(token_scores, dim=2)

                _, max_token_idx = torch.max(tok_max_scores, dim=1)

            importance_token = gates_token.sum(0)

            loss_token = self.cv_squared(importance_token) + self.cv_squared(load_token)


            B, T, D = queries_back.shape
            E = self.num_experts
            y_x_token_list = []
            for i in range(B):
                dispatcher_token = SparseDispatcher_token(self.num_experts, gates_token[i])

                expert_inputs_x_token = dispatcher_token.dispatch(queries_back[i])


                expert_outputs_x_token = [self.experts_token[q](expert_inputs_x_token[q]) for q in
                                          range(self.num_experts)]
                y_x_token = self.dteall_token(
                    self.dte_token(dispatcher_token.combine(expert_outputs_x_token, top_k_logits_token[i])))
                y_x_token = self.proj_token(y_x_token)
                y_x_token_list.append(y_x_token)

            y_x_token = torch.stack(y_x_token_list, dim=0)


            queries = torch.cat(
                [queries_sam[:, :9, :], y_x_token, queries_sam[:, 9:, :]],
                dim=1
            )


        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys,loss_token,max_token_idx


class Attention(nn.Module):


    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        save_attn=False,
        attn=None
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.save_attn = save_attn
        self.attn = attn
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:

        q = self.q_proj(q.to(self.q_proj.weight.dtype))
        k = self.k_proj(k.to(self.k_proj.weight.dtype))
        v = self.v_proj(v.to(self.v_proj.weight.dtype))


        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)


        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        if self.save_attn:
            self.attn = attn.detach()


        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


import torch
import torch.nn.functional as F

@torch.no_grad()
def build_expert_cams_from_block(block, img_hw=None, agg='mean'):


    attn = block.last_cross_attn_attn
    gates = block.last_gates_token
    t_slice = block.last_t_slice
    h, w = (img_hw if img_hw is not None else block.last_hw)

    assert attn is not None and gates is not None and t_slice is not None,\
        "Missing attn/gates/t_slice. Run a forward pass with routing state enabled first."


    a = attn[:, :, t_slice, :].mean(dim=1)


    a = (a - a.min(dim=-1, keepdim=True).values) /\
        (a.max(dim=-1, keepdim=True).values - a.min(dim=-1, keepdim=True).values + 1e-12)

    B, T_back, HW = a.shape
    E = gates.shape[-1]
    cams = []
    for e in range(E):
        mask = (gates[..., e] > 0).float()
        if agg == 'mean':
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            cam_e = (a * mask.unsqueeze(-1)).sum(dim=1) / denom
        elif agg == 'sum':
            cam_e = (a * mask.unsqueeze(-1)).sum(dim=1)
        else:

            cam_e = (a * mask.unsqueeze(-1) + 1e-12).amax(dim=1)

        cam_e = cam_e.reshape(B, h, w)

        cam_e = (cam_e - cam_e.amin(dim=(1,2), keepdim=True)) /\
                (cam_e.amax(dim=(1,2), keepdim=True) - cam_e.amin(dim=(1,2), keepdim=True) + 1e-12)
        cams.append(cam_e)

    cams = torch.stack(cams, dim=1)
    return cams
