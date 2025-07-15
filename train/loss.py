import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, cfg):
        super(FocalLoss, self).__init__()
        self.alpha = cfg.loss.alpha
        self.gamma = cfg.loss.gamma
        self.reduction = cfg.loss.reduction

    def forward(self, score, label):
        p = torch.sigmoid(score)
        label = label.float().view(-1,1)
        ce_loss = F.binary_cross_entropy_with_logits(score, label, reduction='none')
        p_t = p * label + (1-p) * (1-label)
        F_loss = ce_loss * ((1-p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * label + (1-self.alpha) * (1-label)
            F_loss = alpha_t * F_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class CELoss(nn.Module):
    def __init__(self, cfg):
        super(CELoss, self).__init__()
        self.reduction = cfg.loss.reduction

    def forward(self, score, label):
        label = label.view(-1,1)
        ce_loss = F.binary_cross_entropy_with_logits(score, label, reduction='none')
        if self.reduction == 'mean':
            return ce_loss.mean()
        elif self.reduction == 'sum':
            return ce_loss.sum()
        else:
            return ce_loss
        
class L2Loss(nn.Module):
    def __init__(self, cfg):
        super(L2Loss, self).__init__()
        self.reduction = cfg.loss.reduction

    def forward(self, score, label):
        label = label.view(-1,1)
        score = torch.sigmoid(score)
        l2_loss = F.mse_loss(score, label, reduction='none')
        if self.reduction == 'mean':
            return l2_loss.mean()
        elif self.reduction == 'sum':
            return l2_loss.sum()
        else:
            return l2_loss
        
class Evaluator(nn.Module):
    def __init__(self,cfg):
        super(Evaluator, self).__init__()
        self.acceptance_bias = cfg.eval.acceptance_bias
        
    def forward(self, cls_logits, label):
        label = label.view(-1,1)
        prediction = torch.sigmoid(cls_logits)
        bias = abs(label-prediction)
        recall = 0
        if bias <= self.acceptance_bias:
            recall = 1
        return recall
        