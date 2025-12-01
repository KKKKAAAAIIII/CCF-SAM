# model_utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred 是 logits，需要先 sigmoid
        pred = torch.sigmoid(pred)
        
        # 展平张量
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集
        intersection = (pred_flat * target_flat).sum()
        
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss"""
    def __init__(self, weight_dice=0.5, weight_bce=0.5, pos_weight=None):
        super(DiceBCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # pred 是 logits
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算 pt
        pt = torch.exp(-bce_loss)
        
        # 计算 focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 控制假阴性的权重
        self.beta = beta    # 控制假阳性的权重
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 展平
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        
        return 1 - tversky

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        
    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice(pred, target)
        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss



def calculate_pos_weight(data_loader):
    """Calculate positive class weight for handling class imbalance"""
    from tqdm import tqdm
    
    total_pixels = 0
    pos_pixels = 0
    
    print("Calculating class weights...")
    for sample in tqdm(data_loader):
        labels = sample['label']
        total_pixels += labels.numel()
        pos_pixels += labels.sum().item()
    
    neg_pixels = total_pixels - pos_pixels
    pos_weight = neg_pixels / pos_pixels if pos_pixels > 0 else 1.0
    
    print(f"Positive pixels: {pos_pixels} ({pos_pixels/total_pixels*100:.2f}%)")
    print(f"Negative pixels: {neg_pixels} ({neg_pixels/total_pixels*100:.2f}%)")
    print(f"Recommended pos_weight: {pos_weight:.2f}")
    
    return pos_weight