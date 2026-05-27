import torch
import numpy as np
from torch.nn import functional as F
import random
import torch.nn as nn
import os

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):


        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - pred) ** self.gamma
        w_neg = pred ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(pred + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - pred + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):


        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_loss


class MaskMSE(nn.Module):
    def __init__(self, ):
        super(MaskMSE, self).__init__()

    def forward(self, pred, mask, pred_iou):


        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        mse = torch.mean((iou - pred_iou) ** 2)
        return mse


class FocalDice_MSELoss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDice_MSELoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_mse = MaskMSE()
    def forward(self, pred, mask, pred_iou):


        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_mse(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss
