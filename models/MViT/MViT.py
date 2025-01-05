import torch.nn as nn
from models.MViT.TSMixer import *
from models.MViT.TemporalEncoder import *

from einops.layers.torch import Rearrange


class MViT(nn.Module):
    """
    MS-TCT for action detection
    """
    def __init__(self, model_config):
        super(MViT, self).__init__()
        self.in_feat_dim = model_config['in_feat_dim']
        self.inter_channels = model_config['inter_channels']
        self.num_head = model_config['num_head']
        self.mlp_ratio = model_config['mlp_ratio']
        self.num_block = model_config['num_block']
        self.final_embedding_dim = model_config['final_embedding_dim']
        self.num_classes = model_config['num_classes']
        self.patch_size = model_config['patch_size']
        self.image_size = model_config['img_res']
        self.num_patches_1d = self.image_size//self.patch_size


        self.dropout=nn.Dropout()

        self.TemporalEncoder=TemporalEncoder(in_feat_dim=self.in_feat_dim, embed_dims=self.inter_channels,
                 num_head=self.num_head, mlp_ratio=self.mlp_ratio, norm_layer=nn.LayerNorm,num_block=self.num_block)

        self.Temporal_Mixer=Temporal_Mixer(inter_channels=self.inter_channels, embedding_dim=self.final_embedding_dim)

        self.to_temporal_embedding_input = nn.Linear(366, self.in_feat_dim)

        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.in_feat_dim),)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(4 * self.final_embedding_dim),
            nn.Linear(4 * self.final_embedding_dim, self.num_classes * self.patch_size**2)
        )

    def forward(self, x):
        # inputs = self.dropout(inputs)

        # 把TSViT的预处理直接搬过来
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape


        xt = x[:, :, -1, 0, 0]
        x = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.in_feat_dim)
        x = self.to_patch_embedding(x)
        # shape = (24*12*12, 60, 128)
        x = x.reshape(B, -1, T, self.in_feat_dim)
        x += temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.in_feat_dim)

        # Temporal Encoder Module
        x = self.TemporalEncoder(x)

        # Temporal Scale Mixer Module
        x,_ = self.Temporal_Mixer(x)

        # Seg Module
        x = x.mean(dim=-1, keepdim=True)
        x = self.mlp_head(x.reshape(-1, 4 * self.final_embedding_dim))
        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)

        return x 