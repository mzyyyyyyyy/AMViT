import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import einops

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c n -> b n c')
        x = self.norm(x)
        return einops.rearrange(x, 'b n c -> b c n')

class DeformedAgent(nn.Module):
    def __init__(
        self, model_config, n_heads=8, n_head_channels=16, n_groups=1, stride=1, 
        offset_range_factor=1, no_off=False, ksize=9, agent_num=5 # 也先用源代码的默认值，之后要加到 model_config 里
    ):
        super().__init__()
        self.n_groups = n_groups
        self.nc = n_head_channels * n_heads
        self.n_group_channels = self.nc // self.n_groups
        self.ksize = ksize
        self.n_heads = n_heads
        self.n_head_channels = n_head_channels
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        agent_stride = model_config['max_seq_len'] // agent_num
        self.kernel_size = agent_num


        self.conv_offset = nn.Sequential(
            nn.Conv1d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv1d(self.n_group_channels, 1, self.kernel_size, agent_stride, 0, bias=False)
        )

        self.offset_range_factor = offset_range_factor
        self.no_off = no_off

        self.device = model_config['device']

        self.proj_q = nn.Linear(self.nc, self.nc)

        self.cls_token_num = model_config['cls_token_num']

        self.phenology_prior = model_config['phenology_prior']

        self.is_training = False

    def set_mode(self, is_training):
        self.training = is_training
    
    def _get_ref_points(self, L_key, B, device):


        ref = torch.linspace(1, 365, L_key, dtype=int, device=device) # 包含在 1 到 365 之间均匀分布的 L_key 个数。
        x_min, x_max = 1, 365
        y_min, y_max = 0, 59
        ref = (ref - x_min) * (y_max - y_min) / (x_max - x_min) + y_min
        ref = ref[None, :].expand(B * self.n_groups, -1).unsqueeze(-1)  # B * g L 1
        
        return ref


    def forward(self, q, tokens, x_labels=None):
        # 1, 计算偏移量
        # 2，生成参考点（训练和推理时，生成参考点的方式不同）
        # 3，计算位置
        # 4，特征采样
        # 5，原始图像投影得到 agent tokens' q.

        tokens = tokens[:, self.cls_token_num:, :]
        q = q[:, self.cls_token_num:, :]
        B, L, C = tokens.size() # b n d
        device = tokens.device

        q_off = einops.rearrange(q, 'b n (g c) -> (b g) c n', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()
        # b * g 1 ng, 这里 ng 的大小由 conv_offset 模块内部的卷积层决定。

        Lk = offset.size(2)
        n_sample = Lk

        if self.offset_range_factor >= 0 and not self.no_off:
            # offset_range = torch.tensor([1.0 / (Lk - 1.0)], device=self.device).reshape(1, 1, 1) # 创建一个包含值 1.0 / (Lk - 1.0) 的张量，将张量的形状调整为 (1, 1, 1)
            offset = offset.tanh().mul(self.offset_range_factor) # 对 offset 张量应用 tanh 函数，将其值限制在 -1 到 1 之间；将 tanh 结果与 offset_range 相乘，缩放 offset 的值；将结果与 offset_range_factor 相乘，缩放 offset 的值。

        offset = einops.rearrange(offset, 'b p l -> b l p') 
        if self.is_training:
            if torch.cuda.device_count() > 1: # 确保 reference 和 x_lables 在多卡训练时被正确地分配到 GPU 上。
                reference = reference.to('cuda')
                x_labels = x_labels.to('cuda')
                self.phenology_prior = self.phenology_prior.to('cuda')
            for i in range(B):
                reference[i] = self.phenology_prior[x_labels[i]]
        else:
            reference = self._get_ref_points(Lk, B, device)

        if self.no_off:
            offset = offset.fill_(0.0) # 将 offset 置零

        if self.offset_range_factor >= 0:
            pos = offset + reference 
        else:
            pos = (offset + reference).clamp(-1., +1.) # 将张量中的每个元素限制在指定的范围内，如果某个元素小于 -1，则将其设置为 -1；如果某个元素大于 +1，则将其设置为 +1

        if self.no_off:
            x_sampled = F.avg_pool1d(tokens, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Lk, f"Size is {x_sampled.size()}"
        else:
            # 使用线性插值并结合 pos 信息
            pos = pos[..., 0].unsqueeze(1)  # 取 pos 的第一个维度作为插值位置 
            pos = pos.long().expand(B, C, Lk)
            # x_sampled = F.interpolate(
            #     tokens.reshape(B * self.n_groups, self.n_group_channels, L),
            #     size=Lk,
            #     mode='linear',
            #     align_corners=True
            # ) 
            x_sampled = tokens.reshape(B * self.n_groups, self.n_group_channels, L)
            x_sampled = x_sampled.gather(2, pos)  # 使用 pos 进行采样


        x_sampled = x_sampled.reshape(B, n_sample, C)

        q_new = self.proj_q(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample) # shape 与接口处 shape 一致！
        return q_new

class AgentTransformer(nn.Module):
    def __init__(self, dim, num_heads, model_config, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                  **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.temporal_length = model_config['max_seq_len'] + model_config['cls_token_num']
        # 问了一下 GPT 在 swin transformer 中，window_size 的作用。
        self.agent_num = model_config['phenology_num']
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, self.agent_num, self.agent_num))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, self.agent_num, self.agent_num))
        
        # 将 ah aw 替换为 at; 将 ha,wa 替换为 ta.
        self.at_bias = nn.Parameter(torch.zeros(1, num_heads, self.agent_num, self.temporal_length))
        self.ta_bias = nn.Parameter(torch.zeros(1, num_heads, self.temporal_length, self.agent_num))

        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)

        trunc_normal_(self.at_bias, std=.02)
        trunc_normal_(self.ta_bias, std=.02)
        self.pool = nn.AdaptiveAvgPool1d(output_size=self.agent_num)
        self.deformed_agent = DeformedAgent(model_config)

        self.is_training = False

    def set_mode(self, is_training):
        self.is_training = is_training

    def forward(self, x, x_labels=None):
        num, t, d = x.shape
        num_heads = self.num_heads
        head_dim = d // num_heads
        qkv = self.qkv(x).reshape(num, t, 3, d).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c
        # q.shape = (b, n, c)

        # 通过 deformed 方式生成 agent tokens.
        self.deformed_agent.set_mode(self.is_training)
        if self.is_training:
            agent_tokens_q = self.deformed_agent(q, x, x_labels)
        else:
            agent_tokens_q = self.deformed_agent(q, x)
        q = q.reshape(num, t, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(num, t, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(num, t, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens_q = agent_tokens_q.reshape(num, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        # agent.shape = (b, heads, agent_num, c)

        position_bias1 = F.interpolate(self.an_bias, size=self.temporal_length, mode='linear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(num, 1, 1, 1)
        position_bias2 = self.at_bias.repeat(num, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens_q * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.temporal_length, mode='linear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(num, 1, 1, 1)
        agent_bias2 = self.ta_bias.repeat(num, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens_q.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(num, t, d)

        # 暂时删去 DWC 模块
        # v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x