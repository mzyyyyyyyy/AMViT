import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from models.AMViT.AgentTransformer import *
from models.AMViT.MultiTempTransformer import *

class AMViT(nn.Module):
    """
    Temporal-Spatial ViT5 (used in main results, section 4.3)
    For improved training speed, this implementation uses a (365 x dim) temporal position encodings indexed for
    each day of the year. Use TSViT_lookup for a slower, yet more general implementation of lookup position encodings
    """
    def __init__(self, model_config):
        super().__init__()
        self.image_size = model_config['img_res']
        self.patch_size = model_config['patch_size']
        self.num_patches_1d = self.image_size//self.patch_size
        self.num_classes = model_config['num_classes']
        self.num_frames = model_config['max_seq_len']
        self.dim = model_config['dim']
        if 'temporal_depth' in model_config:
            self.temporal_depth = model_config['temporal_depth']
        else:
            self.temporal_depth = model_config['depth']
        if 'spatial_depth' in model_config:
            self.spatial_depth = model_config['spatial_depth']
        else:
            self.spatial_depth = model_config['depth']
        self.heads = model_config['heads']
        self.dim_head = model_config['dim_head']
        self.dropout = model_config['dropout']
        self.emb_dropout = model_config['emb_dropout']
        self.pool = model_config['pool']
        self.scale_dim = model_config['scale_dim']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = self.num_patches_1d ** 2
        patch_dim = (model_config['num_channels'] - 1) * self.patch_size ** 2  # -1 is set to exclude time feature
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(patch_dim, self.dim),)
        self.to_temporal_embedding_input = nn.Linear(366, self.dim)
        self.temporal_token = nn.Parameter(torch.randn(1, self.num_classes, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2)
        )

        self.deform_agent_transformer = AgentTransformer(self.dim, self.heads, model_config, self.heads) # 用 agent attention 的核心代码填充
        self.multi_temporal_transformer = MultiTempTransformer(model_config, self.dim)
        # 用 MSTCT 的核心代码填充
        # 这一模块中的参数暂时先用模块内部预定义的。
        self.fusion = Fusion(model_config) 
        # 先用简单的聚合 cls tokens 实现，然后用 MTV 的核心代码填充
        self.is_training = False

    def set_mode(self, is_training):
        self.is_training = is_training


    def forward(self, x, x_labels=None):

        # 2D CNN tokenizer
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape


        xt = x[:, :, -1, 0, 0]
        x_token = x[:, :, :-1]
        xt = (xt * 365.0001).to(torch.int64)
        xt = F.one_hot(xt, num_classes=366).to(torch.float32)
        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(B, T, self.dim)
        x_token = self.to_patch_embedding(x_token)
        # shape = (24*12*12, 60, 128)
        x_token = x_token.reshape(B, -1, T, self.dim)
        x_token += temporal_pos_embedding.unsqueeze(1)
        x_token = x_token.reshape(-1, T, self.dim)
        cls_temporal_tokens = repeat(self.temporal_token, '() N d -> b N d', b=B * self.num_patches_1d ** 2)

        x_token = torch.cat((cls_temporal_tokens, x_token), dim=1)

        # 将 x_labels 的预处理为与 tokens 对应的 shape，使它们一一对应




        # temporal encoder for AMViT
        self.deform_agent_transformer.set_mode(self.is_training)
        if self.is_training:
            # 将 x_labels 的预处理为与 tokens 对应的 shape，使它们一一对应
            labels = x_labels.view(x_labels.size(0), 1, x_labels.size(1), x_labels.size(2))
            labels_unfolded = F.unfold(labels, kernel_size=self.patch_size, stride=self.patch_size)
            labels_unfolded = labels_unfolded.view(labels.size(0), 1, -1, self.patch_size * self.patch_size)
            labels_mode, _ = torch.mode(labels_unfolded, dim=-1)
            labels_mode = labels_mode.view(-1, 1)   # (B * H * W, 1)

            x = self.deform_agent_transformer(x_token, x_labels)
        else:   
            x = self.deform_agent_transformer(x_token)
        # 这里有一个遗留问题：生成关键物候期 timesteps 的时候，应该去除 cls_tokens. √
        # 还有一个遗留问题，训练和测试的时候，生成关键物候期 timesteps 的方式是不是不同。
        # 还有一个遗留问题，我只用了一个 DAT 模块，是不是应该用整个 DAT 框架，反正输入输出都一样。
        
        x = self.multi_temporal_transformer(x)
        # input = b t d
        # output = 
        # [b t embed_dims[0], 
        # b cls_tokens+(t-cls_tokens//2) embed_dims[1], 
        # b cls_tokens+(t-cls_tokens//4) embed_dims[2], 
        # b cls_tokens+(t-cls_tokens//8) embed_dims[3]]
        # 遗留问题：卷积模块应该只处理非 cls tokens。√
        x = self.fusion(x)
        # (b, cls_tokens, embed_dims[3])
        # 先用最简单的方式实现：只用 b cls_tokens embed_dims[3]

        x = self.dropout(x) # 保留原始 TSViT 的模块，但原因不详。


        # segmentation head
        x = self.mlp_head(x.reshape(-1, self.dim))

        x = x.reshape(B, self.num_classes, self.num_patches_1d**2, self.patch_size**2).permute(0, 2, 3, 1)
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x