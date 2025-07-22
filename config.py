import os
import os.path as osp
import argparse
from datetime import datetime
from easydict import EasyDict as edict

from utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = _C.working_dir
_C.current_time = datetime.now().strftime("%Y%m%d%H%M")
_C.exp_name = f'{osp.basename(_C.working_dir)}_{_C.current_time}'
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.model_dir = osp.join(_C.output_dir, 'models')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.pre_weight_fn = osp.join(_C.root_dir,'weights','geotransformer-3dmatch.pth.tar')
_C.model_fn = osp.join(_C.root_dir,'weights','scorer.pth.tar')

# data
_C.data = edict()
_C.data.dataset_root = osp.join(_C.root_dir, 'data', '3DMatch')

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.025
_C.backbone.dsm_point_num = 128
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.use_weights = True
_C.model.dual_normalization = True
_C.model.group_norm = 32
_C.model.num_classes = 1

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

# train
_C.train = edict()
_C.train.epochs = 30
_C.train.batch_size = 1
_C.train.num_workers = 16
_C.train.point_limit = 30000
_C.train.log_steps = 10#3000
_C.train.val_steps = 10#6000
_C.train.save_steps = 10#3000

_C.optim = edict()
_C.optim.geo_lr = 1e-4 #1e-5
_C.optim.cls_lr = 1e-3 #1e-4
_C.optim.lr_decay_rate = 1 #0.95
_C.optim.lr_decay_step = 1 #the decay step of the learning rate (how many epochs)

_C.loss = edict()
_C.loss.loss_type = 'FocalLoss' # "L2loss", "CEloss"
_C.loss.reduction = 'mean'
_C.loss.alpha = 0.75
_C.loss.gamma = 4.0


def make_cfg():
    return _C
