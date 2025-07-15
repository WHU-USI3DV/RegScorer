import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pairwise_distance import pairwise_distance
from utils.pointcloud import apply_transform_tensor_para
from regscorer.transformer.geotransformer import (
    GeometricTransformer,
)

from regscorer.kpconv.backbone import KPConvFPN

class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

    def forward(self, data_dict):

        # Downsample point clouds
        feats = data_dict['features'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        points_c = data_dict['points'][-1].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]

        # 2. KPFCNN Encoder
        feats_c = self.backbone(feats, data_dict)

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        ) # 1,128,256
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)
        return ref_feats_c_norm,src_feats_c_norm

class Scorer(nn.Module):
    def __init__(self, cfg):
        super(Scorer, self).__init__()
        self.num_classes = cfg.model.num_classes

        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GroupNorm(cfg.model.group_norm,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.GroupNorm(cfg.model.group_norm,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, data_dict, ref_feats_c_norm,src_feats_c_norm):
        ref_length_c = data_dict['lengths'][-1][0].item()
        points_c = data_dict['points'][-1].detach()
        trans = data_dict['trans'] #(n,4,4)

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        src_points_c = apply_transform_tensor_para(src_points_c, trans)

        # normalize
        centroid = torch.mean(ref_points_c, dim=0)
        m = torch.sqrt(torch.sum(ref_points_c**2, dim=1)).max()
        ref_points_c_norm = (ref_points_c - centroid) / m
        src_points_c_norm = (src_points_c - centroid) / m
        ref_points_c_norm = ref_points_c_norm.unsqueeze(0).repeat(trans.shape[0], 1, 1) #(128,3)-->(n,128,3)

        # expand feature matrix
        ref_feats_c_norm = ref_feats_c_norm.unsqueeze(0).repeat(trans.shape[0], 1, 1)
        src_feats_c_norm = src_feats_c_norm.unsqueeze(0).repeat(trans.shape[0], 1, 1)

        # get point's spatial distance and feature similarity
        distance_mat = torch.exp(-pairwise_distance(ref_points_c_norm,src_points_c_norm)) #(n,128,128)
        similarity_mat = torch.matmul(ref_feats_c_norm, src_feats_c_norm.transpose(-1, -2))

        # spatial distance nearest: distance and corresponding feature similarity
        ref2src_nearest_dists, ref2src_nearest_inds = distance_mat.max(dim=-1) #(n,128)
        ref2src_nearest_similarity = torch.gather(similarity_mat,2,ref2src_nearest_inds.unsqueeze(-1)).squeeze(-1)
        src2ref_nearest_dists, src2ref_nearest_inds = distance_mat.max(dim=-2)
        src2ref_nearest_similarity = torch.gather(similarity_mat.transpose(1,2),2,src2ref_nearest_inds.unsqueeze(-1)).squeeze(-1)
        
        nearest_dist = torch.cat([ref2src_nearest_dists,src2ref_nearest_dists],dim=-1) #(n,256)
        nearest_similarity = torch.cat([ref2src_nearest_similarity,src2ref_nearest_similarity],dim=-1)
        sorted_nearest_dist, perm_d = torch.sort(nearest_dist,dim=-1,descending=True)
        sorted_nearest_similarity = torch.gather(nearest_similarity,1,perm_d)
        nearest_score = sorted_nearest_dist * sorted_nearest_similarity

        # max similarity: similarity and corresponding distance
        ref2src_max_similarity, ref2src_maxsimi_inds = similarity_mat.max(dim=-1)
        ref2src_maxsimi_dist = torch.gather(distance_mat,2,ref2src_maxsimi_inds.unsqueeze(-1)).squeeze(-1)
        src2ref_max_similarity, src2ref_maxsimi_inds = similarity_mat.max(dim=-2)
        src2ref_maxsimi_dist = torch.gather(distance_mat.transpose(1,2),2,src2ref_maxsimi_inds.unsqueeze(-1)).squeeze(-1)

        max_similarity = torch.cat([ref2src_max_similarity,src2ref_max_similarity],dim=-1)
        maxsimi_dist = torch.cat([ref2src_maxsimi_dist,src2ref_maxsimi_dist],dim=-1)
        sorted_max_similarity, perm_f = torch.sort(max_similarity,dim=-1,descending=True)
        sorted_maxsimi_dist = torch.gather(maxsimi_dist,1,perm_f)
        maxsimi_score = sorted_max_similarity * sorted_maxsimi_dist

        # classifier
        joint_feature = torch.cat([nearest_score,maxsimi_score],dim=-1) #(n,512)
        cls_logits = self.cls_head(joint_feature)
        return cls_logits


def create_geo_model(config):
    model = GeoTransformer(config)
    return model

def create_score_model(config):
    model = Scorer(config)
    return model

def main():
    from config import make_cfg
    cfg = make_cfg()
    geo_model = create_geo_model(cfg)
    score_model = create_score_model(cfg)
    print(geo_model.state_dict().keys())
    print(geo_model)
    print(score_model.state_dict().keys())
    print(score_model)


if __name__ == '__main__':
    main()
