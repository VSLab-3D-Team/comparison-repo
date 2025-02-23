from src.model.model_utils.network_PointNet_512 import PointNetEncoder
from src.model.model_utils.networks_base import BaseNetwork
from src.model.model_utils.network_util import Gen_Index
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x  # Skip Connection
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # Add Skip Connection
        out = self.activation(out)
        return out

class RelFeatNaiveExtractor(nn.Module):
    def __init__(self, input_dim, geo_dim, out_dim, num_layers=6):
        super(RelFeatNaiveExtractor, self).__init__()
        self.obj_proj = nn.Linear(input_dim, 512)
        self.geo_proj = nn.Linear(geo_dim, 512)
        self.merge_layer = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding="same")
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_layers)])
        self.fc_out = nn.Linear(512, out_dim)  # 출력 레이어
        

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, geo_feats: torch.Tensor):
        # All B X N_feat size
        p_i, p_j, g_ij = self.obj_proj(x_i), self.obj_proj(x_j), self.geo_proj(geo_feats)
        m_ij = torch.cat([
            p_i.unsqueeze(1), p_j.unsqueeze(1), g_ij.unsqueeze(1)
        ], dim=1)
        
        e_ij = self.merge_layer(m_ij).squeeze(1) # B X 512
        r_ij = self.res_blocks(e_ij)
        return self.fc_out(r_ij)

class BFeatRelObjConNet(BaseNetwork):
    def __init__(self, config, device):
        super(BFeatRelObjConNet, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.p_model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.point_encoder = PointNetEncoder(device, channel=9)
        # self.point_encoder.load_state_dict(torch.load(self.t_config.ckp_path))
        # self.point_encoder = self.point_encoder.to(self.device).eval()
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.relation_encoder = RelFeatNaiveExtractor(
            self.m_config.dim_obj_feats,
            self.m_config.dim_geo_feats,
            self.m_config.dim_edge_feats,
            self.m_config.num_layers
        ).to(self.device)
        
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        is_train = True
    ):
        if is_train:
            bsz = obj_pts.shape[0] // 2
            obj_feats_cat, _, _ = self.point_encoder(obj_pts)
            obj_t1_feats = obj_feats_cat[:bsz, ...]
            obj_t2_feats = obj_feats_cat[bsz:, ...]
            
            obj_feats = torch.stack([obj_t1_feats, obj_t2_feats]).mean(dim=0)
        else:
            obj_feats, _, _ = self.point_encoder(obj_pts)
        
        x_i_feats, x_j_feats = self.index_get(obj_feats, edge_indices)
        geo_i_feats, geo_j_feats = self.index_get(descriptor, edge_indices)
        edge_feats = self.relation_encoder(x_i_feats, x_j_feats, geo_i_feats - geo_j_feats)
        
        if is_train:
            return obj_feats, edge_feats, obj_t1_feats, obj_t2_feats
        else:
            return obj_feats, edge_feats