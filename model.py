import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataloaders.data_utils import get_points_from_mask, get_bboxes_from_mask
from transformers import AutoTokenizer
import pickle
import re
import random


class SegMoTE(nn.Module):
    def __init__(
        self,
        sam,
        test_mode=False,
        multimask_output=True,
        category_weights=None,
        select_mask_num=None,
        text_tokenizer_name="openai/clip-vit-base-patch32",
        ):
        super().__init__()

        self.device = sam.device
        self.image_encoder = sam.image_encoder
        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder
        self.text_model = sam.text_model
        self.text_out_dim = sam.text_out_dim
        self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        self.test_mode = test_mode
        self.multimask_output = multimask_output
        self.category_weights = category_weights
        self.select_mask_num = select_mask_num

        self.image_format = sam.image_format
        self.image_size = sam.prompt_encoder.input_image_size


        for n, value in self.text_model.named_parameters():
            value.requires_grad = False

        if category_weights is not None:
            self.load_category_weights(category_weights)

    def image_forward(self, image):
        img_shape = image.shape
        image_embedding,interm_embeddings = self.image_encoder(image)
        assert len(image_embedding.shape) == 4, f'required shape is (B, C, H, W), but we get {image_embedding.shape}'

        if self.test_mode:
            return_img_embed = image_embedding
            return_interm_embed = interm_embeddings[0].detach().clone()
        else:
            interm_embeddings_repeat = []
            interm_embedding = interm_embeddings[0].detach().clone()

            image_embeddings_repeat = []
            image_embedding = image_embedding.detach().clone()
            for bs in range(img_shape[0]):
                interm_embed = interm_embedding[bs].repeat(self.select_mask_num, 1, 1, 1)
                interm_embeddings_repeat.append(interm_embed)

                image_embed = image_embedding[bs].repeat(self.select_mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            return_interm_embed = torch.cat(interm_embeddings_repeat, dim=0).to(interm_embedding.device)
            return_img_embed = torch.cat(image_embeddings_repeat, dim=0).to(image_embedding.device)


        return return_img_embed,return_interm_embed

    def forward_decoder(self, image_embedding, prompt,interm_embeddings,test_mode=False,inter=False,if_prompt=None):
        if  prompt.get("point_coords", None) is None:
            points = None
        else:
            points = (prompt["point_coords"], prompt["point_labels"])

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=prompt.get("bboxes", None),
            masks=prompt.get("mask_inputs", None),
            text=prompt.get("text_inputs", None),
        )

        outputs,loss,moe_idx = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            text_prompt_embeddings=prompt.get("text_inputs", None),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            interm_embeddings=interm_embeddings,
            inter=inter,
            if_prompt=if_prompt,
            test_mode=test_mode,
          )

        if self.multimask_output:
            low_res_masks, iou_pred, semantic_pred = self.get_max_pred(outputs,inter)

        else:

            low_res_masks, iou_pred, semantic_pred = outputs['low_res_masks'], outputs['iou_pred'], outputs[
                'semantic_pred']

        masks = F.interpolate(low_res_masks, size=self.image_size, mode='bilinear', align_corners=False)

        outputs = {
            'masks': masks.float(),
            'low_res_masks': low_res_masks,
            'iou_pred': iou_pred,
            'semantic_pred': semantic_pred,
        }
        return outputs,loss,moe_idx

    def forward(self, image, prompt):
        image_embedding, interm_embeddings = self.image_forward(image)
        return self.forward_decoder(
            image_embedding,
            prompt,
            interm_embeddings=interm_embeddings,
            test_mode=self.test_mode,
        )

    def supervised_prompts(self, classes, labels, pred_masks, low_res_masks, specify_prompt):
        bs_prompts = {}
        if low_res_masks is not None:
            bs_prompts.update(self.process_mask_prompt(low_res_masks))
        if specify_prompt == 'points':
            bs = labels.shape[0]
            bs_prompts.update(self.process_points_prompt(bs, labels, pred_masks))
        elif specify_prompt == 'text':
            bs_prompts.update(self.process_text_prompt(classes))
        elif specify_prompt == 'bboxes':
            bs = labels.shape[0]
            bs_prompts.update(self.process_bboxes_prompt(bs, labels))

        assert len(bs_prompts)>0, f'prompt error: {bs_prompts}'
        return bs_prompts

    def unsupervised_prompts(self, pseudo_labels, pred_masks, low_res_masks, specify_prompt):
        bs_prompts = {}
        if low_res_masks is not None:
            bs_prompts.update(self.process_mask_prompt(low_res_masks))
        if specify_prompt == 'points':
            bs = pseudo_labels.shape[0]
            bs_prompts.update(self.process_points_prompt(bs, pseudo_labels, pred_masks))

        elif specify_prompt == 'bboxes':
            bs = pseudo_labels.shape[0]
            bs_prompts.update(self.process_bboxes_prompt(bs, pseudo_labels))

        assert len(bs_prompts)>0, f'prompt error: {bs_prompts}'
        return bs_prompts

    def process_text_prompt(self, classes):
        bs_text_prompt = self.text_tokenizer(classes)
        return {'text_inputs': bs_text_prompt.to(self.device)}

    def process_bboxes_prompt(self, bs, labels):
        bs_bboxes = [get_bboxes_from_mask(labels[idx]) for idx in range(bs)]
        return {'bboxes': torch.stack(bs_bboxes, dim=0).to(self.device)}

    def process_points_prompt(self, bs, labels, pred_masks=None):
        if self.test_mode:
            point_num = 1
        else:
            point_num = random.choice([1,3,4,7])

        if pred_masks is not None:
            pred_masks = torch.sigmoid(pred_masks)
            pred_masks = (pred_masks > 0.5).bool().squeeze(1)

        labels = labels.bool().squeeze(1)
        error_area = pred_masks ^ labels

        bs_point_coords = torch.empty((bs, point_num, 2), dtype=torch.long, device=labels.device)
        bs_point_labels = torch.empty((bs, point_num), dtype=torch.long, device=labels.device)

        for idx in range(bs):
            if pred_masks is None:
                point_coords, point_labels = get_points_from_mask(labels[idx], get_point=1)
            else:
                point_coords, point_labels = self.get_points_from_interaction(error_area[idx], pred_masks[idx], labels[idx], get_point=point_num)

            bs_point_coords[idx, :] = torch.as_tensor(point_coords, device=labels.device)
            bs_point_labels[idx, :] = torch.as_tensor(point_labels, device=labels.device)

        return {
            'point_coords': bs_point_coords,
            'point_labels': bs_point_labels
        }

    def process_mask_prompt(self, low_res_masks):
        low_res_masks_logist = low_res_masks.detach().clone()

        return {'mask_inputs': low_res_masks_logist.to(self.device)}

    def text_tokenizer(self, text, tamplate='A segmentation area of a {}.'):
        norm_text = []
        for t in text:
            t = self.categories_map[t][0]
            t = t.lower().replace('_', ' ').replace("-", " ")
            t = re.sub(r'\s+', ' ', t)
            norm_text.append(t)
        text_list = [tamplate.format(t) for t in norm_text]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        for key in tokens.keys():
            tokens[key] = tokens[key].to(self.device)
        text_outputs = self.text_model(**tokens)
        text_embedding = text_outputs.pooler_output
        text_embedding = self.text_out_dim(text_embedding)
        return text_embedding

    def load_category_weights(self, src_weights=None):
        if src_weights is not None:
            with open(src_weights, "rb") as f:
                self.src_weights, self.categories_map, self.category_to_index, self.index_to_category = pickle.load(f)
                self.src_weights = torch.tensor(self.src_weights).to(self.device)

    def category_labels(self, classes):
        norm_target = []
        for clas in classes:
            clas = self.categories_map[clas][1]
            category = clas.lower().replace('_', ' ').replace("-", " ")
            category = category.replace('left','').replace('right', '').strip()
            category = re.sub(r'\s+', ' ', category)
            norm_target.append(category)
        return torch.tensor([self.category_to_index[clas] for clas in norm_target]).unsqueeze(-1).to(self.device)

    def category_loss(self, semantic_preds, classes, ce_loss):
        labels = self.category_labels(classes)
        logits = nn.functional.normalize(semantic_preds, dim=-1) @ self.src_weights
        probs = nn.functional.softmax(logits, dim=-1)
        loss = ce_loss(probs.squeeze(1), labels.squeeze(1))
        return loss, probs

    def get_max_pred(self, outputs,inter):
        low_res_masks = outputs['low_res_masks']
        iou_pred = outputs['iou_pred']
        semantic_pred = outputs['semantic_pred']
        candidate_masks = outputs['candidate_masks']
        max_values, max_indices = torch.max(iou_pred, dim=1, keepdim=True)


        low_res_masks_selected = low_res_masks.unsqueeze(1)
        if inter:
            low_mask_indices = max_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, candidate_masks.shape[2], candidate_masks.shape[3]
            )
            low_res_masks_selected = torch.gather(candidate_masks, 1, low_mask_indices)
        semantic_idices = max_indices.unsqueeze(-1).expand(-1, -1, 512)
        semantic_selected = torch.gather(semantic_pred, 1, semantic_idices)
        return low_res_masks_selected, max_values, semantic_selected

    def get_points_from_interaction(self, error, pr, gt, get_point=1):
        pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
        error = error.cpu().numpy()
        indices = np.argwhere(error == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], get_point, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(get_point, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], get_point, replace=True)]

        selected_indices = selected_indices.reshape(-1, 2)
        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if pred[x,y] == 0 and gt[x,y] == 1:
                label = 1
            elif pred[x,y] == 1 and gt[x,y] == 0:
                label = 0
            else:
                label = -1
            points.append((y, x))
            labels.append(label)
        return np.array(points), np.array(labels)

if __name__ == '__main__':
    print('Test Network')
