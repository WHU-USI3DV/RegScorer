import numpy as np
import torch
import argparse
from tqdm import tqdm
import time,os
import open3d as o3d

from utils.data import precompute_data_stack_mode
from utils.torch import to_cuda
from regscorer.model import create_geo_model,create_score_model
from config import make_cfg

def load_snapshot(geo_model,score_model, snapshot):
    print('Loading from "{}".'.format(snapshot))
    state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
    assert 'model' in state_dict, 'No model can be loaded.'
    geo_params,score_params = {},{}
    for key, value in state_dict['model'].items():
        if key.startswith('backbone.') or key.startswith('transformer.'):
            geo_params[key] = value
        elif key.startswith('cls_head.'):
            score_params[key] = value
    geo_model.load_state_dict(geo_params, strict=True)
    score_model.load_state_dict(score_params, strict=True)
    print('Model has been loaded.')

class regscorer:
    def __init__(self,cfg):
        self.cfg = cfg
        self.point_limit = 30000
        self.neighbor_limits = np.array([41, 36, 34, 15])
        self.voxel_size = cfg.score_voxel
        model_cfg = make_cfg()
        self.geo_model = create_geo_model(model_cfg).cuda()
        self.score_model = create_score_model(model_cfg).cuda()
        load_snapshot(self.geo_model,self.score_model,self.cfg.weight_fn)
        self.geo_model.eval()
        self.score_model.eval()

    def dict_pre(self,data_dict):
        collated_dict = {}
        # array to tensor
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

        # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
        feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
        points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
        lengths = torch.LongTensor([points.shape[0] for points in points_list])
        points = torch.cat(points_list, dim=0)
        # remove wrapping brackets
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]
        collated_dict['features'] = feats
        input_dict = precompute_data_stack_mode(points, lengths, num_stages=4, voxel_size=self.voxel_size, 
                                                radius=0.0625, neighbor_limits = self.neighbor_limits, point_num=128)
        collated_dict.update(input_dict)
        return(collated_dict)

    def score(self, pc0, pc1, trans):
        pc0 = pc0.voxel_down_sample(self.voxel_size)
        pcd0 = np.array(pc0.points)
        if pcd0.shape[0] > self.point_limit:
            indices = np.random.permutation(pcd0.shape[0])[: self.point_limit]
            pcd0 = pcd0[indices]
        pc1 = pc1.voxel_down_sample(self.voxel_size)
        pcd1 = np.array(pc1.points)
        if pcd1.shape[0] > self.point_limit:
            indices = np.random.permutation(pcd1.shape[0])[: self.point_limit]
            pcd1 = pcd1[indices]

        data_dict = {}
        data_dict['ref_points'] = pcd0.astype(np.float32)
        data_dict['src_points'] = pcd1.astype(np.float32)
        data_dict['ref_feats'] = np.ones((pcd0.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((pcd1.shape[0], 1), dtype=np.float32)
        collated_dict = self.dict_pre(data_dict)
        collated_dict['trans'] = torch.from_numpy(trans)
        collated_dict = to_cuda(collated_dict)
        ref_feats_c_norm,src_feats_c_norm = self.geo_model(collated_dict)

        cls_logits = self.score_model(collated_dict,ref_feats_c_norm,src_feats_c_norm)
        scores = torch.sigmoid(cls_logits.squeeze(-1)).detach().cpu().numpy()
        torch.cuda.empty_cache()
        np.save(f'{self.cfg.save_fn}',scores)



parser = argparse.ArgumentParser()
parser.add_argument('--ref_pc_fn',default='./data/demo/cloud_bin_0.ply',type=str, help='Reference point cloud file')
parser.add_argument('--src_pc_fn',default='./data/demo/cloud_bin_1.ply',type=str, help='Source point cloud file')
parser.add_argument('--trans_fn',default='./data/demo/0-1.npz',type=str, help='Candidate transformations file')
parser.add_argument('--save_fn',default='./data/demo/scores_0-1.npy',type=str, help='Output scores file')
parser.add_argument('--weight_fn',default='./weights/epoch-2.pth.tar',type=str, help='Pretrained weights file')
parser.add_argument('--score_voxel',default=0.025,type=float)
config = parser.parse_args()

scorer = regscorer(config)

candidates = np.load(config.trans_fn, allow_pickle=True)['candidates']
trans = []
for i in range(len(candidates)):
    trans.append(candidates[i]['trans'])
trans = np.array(trans)
print(trans.shape)
pc0 = o3d.io.read_point_cloud(config.ref_pc_fn)
pc1 = o3d.io.read_point_cloud(config.src_pc_fn)
scorer.score(pc0,pc1,trans)
