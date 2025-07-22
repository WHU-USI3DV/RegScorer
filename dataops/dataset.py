import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data


class ThreeDMatchPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
    ):
        super(ThreeDMatchPairDataset, self).__init__()

        self.data_root = dataset_root
        self.metadata_root = osp.join(self.data_root, 'metadata')

        self.subset = subset
        self.point_limit = point_limit

        with open(osp.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        points = torch.load(osp.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def _load_train_trans(self, file_name):
        all_trans = np.load(osp.join(self.data_root, file_name),allow_pickle=True)['all_trans']
        trans = []
        labels = []
        for i in range(len(all_trans)):
            tran = all_trans[i]['trans']
            label = all_trans[i]['label']
            trans.append(tran)
            labels.append(label)
        trans = np.array(trans).astype(np.float32)
        labels = np.array(labels).astype(np.int32)
        return trans,labels

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']

        # get point cloud
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])
        trans,labels = self._load_train_trans(metadata['trans'])


        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)

        data_dict['trans'] = torch.from_numpy(trans)
        data_dict['labels'] = torch.from_numpy(labels)

        return data_dict