import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC
from utils.torch import to_cuda

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
        if cfg.loss.loss_type == 'FocalLoss':
            self.loss = FocalLoss(cfg).cuda()
        elif cfg.loss.loss_type == 'CEloss':
            self.loss = CELoss(cfg).cuda()
        elif cfg.loss.loss_type == 'L2loss':
            self.loss = L2Loss(cfg).cuda()
        
    def forward(self, geo_model,score_model,val_loader):
        geo_model.eval()
        score_model.eval()
        all_auc = []
        all_loss = []
        for data_dict in tqdm(val_loader):
            data_dict = to_cuda(data_dict)
            ref_feats_c_norm,src_feats_c_norm = geo_model(data_dict)
            cls_logits = score_model(data_dict,ref_feats_c_norm,src_feats_c_norm)
            labels = data_dict['labels']
            loss = self.loss(cls_logits,labels)
            all_loss.append(loss)
            metric = BinaryAUROC().to('cuda')
            metric.update(cls_logits,labels)
            auroc = metric.compute()
            all_auc.append(auroc)
        val_loss = torch.mean(torch.tensor(all_loss))
        val_auc = torch.mean(torch.tensor(all_auc))
        return val_loss,val_auc
        