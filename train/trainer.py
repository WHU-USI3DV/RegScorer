import os
import os.path as osp
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from collections import OrderedDict

from regscorer.model import create_geo_model,create_score_model
from train.loss import FocalLoss,CELoss,L2Loss,Evaluator
from dataops.dataset import ThreeDMatchPairDataset
from utils.logger import Logger
from utils.common import ExpDecayLR,reset_learning_rate
from utils.torch import reset_seed_worker_init_fn,to_cuda
from utils.data import calibrate_neighbors_stack_mode,registration_collate_fn_stack_mode


class Trainer():
    def __init__(self,cfg):
        self.cfg = cfg
        self.epochs = self.cfg.train.epochs
        # init logger
        log_file = osp.join(cfg.log_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
        self.logger = Logger(log_file=log_file)

        self._init_dataset()
        self._init_network()
        self._load_geo_weight()

        self.best_para = 0
        self.start_step = 0

    def _init_dataset(self):
        start_time = time.time()
        self.train_set = ThreeDMatchPairDataset(
            dataset_root=self.cfg.data.dataset_root,
            subset='train_trans',
            point_limit=self.cfg.train.point_limit)
        self.val_set = ThreeDMatchPairDataset(
            dataset_root=self.cfg.data.dataset_root,
            subset='val_trans',
            point_limit=self.cfg.train.point_limit) 

        self.neighbor_limits = calibrate_neighbors_stack_mode(
            self.train_set,
            registration_collate_fn_stack_mode,
            self.cfg.backbone.num_stages,
            self.cfg.backbone.init_voxel_size,
            self.cfg.backbone.init_radius,
            self.cfg.backbone.dsm_point_num)

        self.train_loader= DataLoader(
            self.train_set,
            self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            collate_fn=partial(
                registration_collate_fn_stack_mode,
                num_stages=self.cfg.backbone.num_stages,
                voxel_size=self.cfg.backbone.init_voxel_size,
                search_radius=self.cfg.backbone.init_radius,
                neighbor_limits=self.neighbor_limits,
                point_num=self.cfg.backbone.dsm_point_num,
                precompute_data=True,
            ),
            worker_init_fn=reset_seed_worker_init_fn,
            pin_memory=False,
            drop_last=False,
        )
        self.val_loader= DataLoader(
            self.val_set,
            self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            collate_fn=partial(
                registration_collate_fn_stack_mode,
                num_stages=self.cfg.backbone.num_stages,
                voxel_size=self.cfg.backbone.init_voxel_size,
                search_radius=self.cfg.backbone.init_radius,
                neighbor_limits=self.neighbor_limits,
                point_num=self.cfg.backbone.dsm_point_num,
                precompute_data=True,
            ),
            worker_init_fn=reset_seed_worker_init_fn,
            pin_memory=False,
            drop_last=False,
        )
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.\n'.format(loading_time)
        message += f'train set len {len(self.train_loader)}\nval set len {len(self.val_loader)}\n'
        message += 'Calibrate neighbors: {}.'.format(self.neighbor_limits)
        self.logger.info(message)
        
    def _init_network(self):
        # model
        self.geo_model = create_geo_model(self.cfg).cuda()
        self.score_model = create_score_model(self.cfg).cuda()
        # optimizer
        self.optimizer_geo = Adam([
            {'params':self.geo_model.backbone.parameters(), 'lr':self.cfg.optim.geo_lr},
            {'params':self.geo_model.transformer.parameters(), 'lr':self.cfg.optim.geo_lr}])
        self.optimizer_cls = Adam([
            {'params':self.score_model.cls_head.parameters(), 'lr':self.cfg.optim.cls_lr}])
        # scheduler
        self.lr_setter_geo=ExpDecayLR(self.cfg.optim.geo_lr, self.cfg.optim.lr_decay_rate, len(self.train_loader)*self.cfg.optim.lr_decay_step)
        self.lr_setter_cls=ExpDecayLR(self.cfg.optim.cls_lr, self.cfg.optim.lr_decay_rate, len(self.train_loader)*self.cfg.optim.lr_decay_step)
        # lcs function, evaluator
        if self.cfg.loss.loss_type == 'FocalLoss':
            self.loss = FocalLoss(self.cfg).cuda()
        elif self.cfg.loss.loss_type == 'CEloss':
            self.loss = CELoss(self.cfg).cuda()
        elif self.cfg.loss.loss_type == 'L2loss':
            self.loss = L2Loss(self.cfg).cuda()
        self.evaluator = Evaluator(self.cfg).cuda()

    def _load_geo_weight(self, fix_prefix=True):
        if os.path.exists(self.cfg.pre_weight_fn): 
            self.logger.info('Loading from "{}".'.format(self.cfg.pre_weight_fn))
            state_dict = torch.load(self.cfg.pre_weight_fn, map_location=torch.device('cpu'))

            # Load model
            model_dict = state_dict['model']
            layers_to_remove = ['backbone.decoder3.mlp.weight','backbone.decoder3.mlp.bias',
                                'backbone.decoder3.norm.norm.weight', 'backbone.decoder3.norm.norm.bias',
                                'backbone.decoder2.mlp.weight', 'backbone.decoder2.mlp.bias',
                                'optimal_transport.alpha']
            for key in layers_to_remove:
                model_dict.pop(key,None)
            self.geo_model.load_state_dict(model_dict, strict=True)

            # log missing keys and unexpected keys
            snapshot_keys = set(model_dict.keys())
            model_keys = set(self.geo_model.state_dict().keys())
            missing_keys = model_keys - snapshot_keys
            unexpected_keys = snapshot_keys - model_keys
            if len(missing_keys) > 0:
                message = f'Missing keys: {missing_keys}'
                self.logger.warning(message)
            if len(unexpected_keys) > 0:
                message = f'Unexpected keys: {unexpected_keys}'
                self.logger.warning(message)
            self.logger.info('GEO_pretrained model has been loaded.')

    def _load_model(self):
        if os.path.exists(self.cfg.model_fn): 
            checkpoint = torch.load(self.cfg.model_fn)
            self.geo_model.load_state_dict(checkpoint['geo_model_state_dict'], strict=True)
            self.score_model.load_state_dict(checkpoint['score_model_state_dict'], strict=True)
            self.optimizer_geo.load_state_dict(checkpoint['geo_optimizer_state_dict'])
            self.optimizer_cls.load_state_dict(checkpoint['cls_optimizer_state_dict'])
            self.best_para = checkpoint['best_para']
            self.start_step = checkpoint['start_step']
            print(f'==> resuming from step {self.start_step} best para {self.best_para}')

    def _save_model(self, step, save_fn=None):
        save_fn=self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step':step,
            'best_para':self.best_para,
            'geo_model_state_dict': self.geo_model.state_dict(),
            'score_model_state_dict': self.score_model.state_dict(),
            'geo_optimizer_state_dict': self.optimizer_geo.state_dict(),
            'cls_optimizer_state_dict': self.optimizer_cls.state_dict(),
        },save_fn)


    def run(self):
        self._load_model()
        pbar=tqdm(total=self.epochs*len(self.train_loader),bar_format='{r_bar}')
        pbar.update(self.start_step)
        step=self.start_step
        whole_loss=0
        start_epoch=self.start_step//len(self.train_loader)
        self.start_step=self.start_step-start_epoch*len(self.train_loader)
        whole_step=len(self.train_loader)*self.epochs
        for epoch in range(start_epoch,self.epochs):
            for i,data_dict in enumerate(self.train_loader):
                step+=1
                data_dict = to_cuda(data_dict)
                self.geo_model.train()
                self.score_model.train()
                reset_learning_rate(self.optimizer_geo,self.lr_setter_geo(step))
                reset_learning_rate(self.optimizer_cls,self.lr_setter_cls(step))
                self.optimizer_geo.zero_grad()
                self.optimizer_cls.zero_grad()
                self.geo_model.zero_grad()
                self.score_model.zero_grad()

                ref_feats_c_norm,src_feats_c_norm = self.geo_model(data_dict)
                cls_logits = self.score_model(data_dict,ref_feats_c_norm,src_feats_c_norm)
                labels = data_dict['labels']

                loss=self.loss(cls_logits,labels)
                loss.backward()
                self.optimizer_geo.step()
                self.optimizer_cls.step()

                whole_loss += loss.detach()
                if (step+1) % self.cfg.train.log_steps == 0:
                    loss_info=f'Train-step{step+1}-loss:{whole_loss/self.cfg.train.log_steps}'
                    self.logger.info(loss_info)
                    whole_loss=0

                if (step+1) % self.cfg.train.val_steps == 0:
                    val_loss,val_auc=self.evaluator(self.geo_model, self.score_model, self.val_loader)
                    print(f'validation auc now: {val_auc:.5f}')
                    if val_auc >= self.best_para:
                        print(f'best validation auc now: {val_auc:.5f} previous {self.best_para:.5f}')
                        self.best_para=val_auc
                        best_pth_fn = f'{self.cfg.model_dir}/epoch{epoch}-step{step+1}-best_auc{val_auc:.3f}.pth.tar'
                        self._save_model(step+1,best_pth_fn)
                    val_info = f'Val-step{step+1}-loss:{val_loss}-auc:{val_auc}'
                    self.logger.info(val_info)

                if (step+1)%self.cfg.train.save_steps==0:
                    pth_fn = f'{self.cfg.model_dir}/epoch{epoch}-model.pth.tar'
                    self._save_model(step+1,pth_fn)

                pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),
                                 lr_geo=self.optimizer_geo.state_dict()['param_groups'][0]['lr'],
                                 lr_cls=self.optimizer_cls.state_dict()['param_groups'][0]['lr'],)
                pbar.update(1)
                if step>=whole_step:
                    break
            if step>=whole_step:
                    break
        pbar.close()