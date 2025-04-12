import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

def get_2d_sincos_pos_embed(embed_dim, height, width):
    """
    embed_dim: int, 嵌入维度（必须为偶数）
    height, width: int, 特征图的高度和宽度
    返回 shape: (H*W, embed_dim)
    """
    # 生成二维网格
    grid_y = torch.arange(height, dtype=torch.float32)
    grid_x = torch.arange(width, dtype=torch.float32)
    # 使用 meshgrid 注意新版 PyTorch 建议指定 indexing='ij'
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')  # shape: (height, width)
    grid = torch.stack([grid_y, grid_x], dim=-1)  # shape: (height, width, 2)
    grid = grid.view(-1, 2)  # shape: (H*W, 2)

    assert embed_dim % 2 == 0, "嵌入维度必须为偶数"
    dim_half = embed_dim // 2
    pos_embed = torch.zeros((grid.shape[0], embed_dim))

    # 为两部分分别计算 sin-cos 编码（对 y 和 x 方向分别处理）
    # 定义分母项（类似于 Transformer 原论文中的实现）
    div_term = torch.exp(torch.arange(0, dim_half, 2, dtype=torch.float32) * -(math.log(10000.0) / dim_half))
    
    # 对 y 坐标
    pos_embed[:, 0:dim_half:2] = torch.sin(grid[:, 0:1] * div_term)
    pos_embed[:, 1:dim_half:2] = torch.cos(grid[:, 0:1] * div_term)
    # 对 x 坐标
    pos_embed[:, dim_half::2] = torch.sin(grid[:, 1:2] * div_term)
    pos_embed[:, dim_half+1::2] = torch.cos(grid[:, 1:2] * div_term)
    
    return pos_embed  # (H*W, embed_dim)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, x):
        # x:(N, C, H, W)
        N, C, H, W = x.shape
        # 将特征图划分为不重叠的窗口
        x = x.view(N, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()# (N, H//window_size, W//window_size, window_size, window_size, C)
        
        # 对每个窗口应用自注意力
        x = x.transpose(0, 1) # （Ws[0]*Ws[1], N*nums_windows, C）
        x, _ = self.attn(x, x, x)  # (Ws[0]*Ws[1], N*nums_windows, C)
        x = x.transpose(0, 1)  # (N*nums_windows, Ws[0]*Ws[1], C)
        
        # 将窗口的输出重新组合成原始形状
        x = x.view(N, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous() # (N, C, H//window_size, window_size, W//window_size, window_size)
        x = x.view(N, C, H, W)  # (N, C, H, W)
        return x
    
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: (L, N, C)
        L, N, C = x.shape
        
        # 投影并重塑
        q = self.q_proj(x).reshape(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (N, H, L, D)
        k = self.k_proj(x).reshape(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (N, H, L, D)
        v = self.v_proj(x).reshape(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # (N, H, L, D)
        
        # 高效注意力：对键和值进行全局聚合
        k_global = k.mean(dim=2, keepdim=True)  # (N, H, 1, D)
        v_global = v.mean(dim=2)  # (N, H, D)
        
        attn = (q @ k_global.transpose(-2, -1)) * self.scale  # (N, H, L, 1)
        attn = F.softmax(attn, dim=-1)  # (N, H, L, 1)
        
        # 应用注意力
        out = attn @ v_global.unsqueeze(2)  # (N, H, L, D)
        
        # 重塑和投影
        out = out.permute(2, 0, 1, 3).reshape(L, N, C)  # (L, N, C)
        out = self.out_proj(out)  # (L, N, C)
        return out
        

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, efficient_attn=True):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        
        if efficient_attn:
            self.attn = EfficientAttention(embed_dim, num_heads)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (L, N, C)
        x2 = self.norm1(x)
        
        if isinstance(self.attn, nn.MultiheadAttention):
            attn_output, _ = self.attn(x2, x2, x2)
        else:
            attn_output = self.attn(x2)
            
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x
    
class TransformerModule(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.0, efficient_attn=True):
        super(TransformerModule, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(input_dim, num_heads, mlp_ratio, dropout, efficient_attn)
            for _ in range(num_layers)
        ])
        self.input_dim = input_dim

    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        L = H * W
        # 将空间展平并转置为 (L, N, C)，其中 L = H * W
        x = x.view(N, C, L).permute(2, 0, 1)
        
        # 生成固定的 sin-cos 位置编码
        pos_embed = get_2d_sincos_pos_embed(C, H, W) # (L, C)
        pos_embed = pos_embed.unsqueeze(1).to(x.device)  #  (L, 1, C)
        x = x + pos_embed  # 添加位置编码
        
        
        
        for layer in self.layers:
            x = layer(x)
        # 将序列 reshape 回 (N, C, H, W)
        x = x.permute(1, 2, 0).view(N, C, H, W)
        return x

class Decoder(nn.Module):
    """原始DeepLab V3+解码器，不含Transformer"""
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        # 1x1降维低级特征
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        # 原始的卷积解码器
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )
        
        self._init_weight()

    def forward(self, x, low_level_feat):
        # 处理低级特征
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        # 上采样高层特征以匹配低级特征尺寸
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        # 拼接特征
        x = torch.cat((x, low_level_feat), dim=1)
        
        # 应用卷积层
        x = self.last_conv(x)
        
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TransformerDecoder(nn.Module):
    """使用Transformer增强的解码器"""
    def __init__(self, num_classes, backbone, BatchNorm, transformer_config):
        super(TransformerDecoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        # 1x1降维低级特征
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        # 确保配置参数存在
        if transformer_config is None:
            transformer_config = {
                'num_heads': 4,
                'num_layers': 2,
                'mlp_ratio': 4.0,
                'dropout': 0.1,
                'efficient_attn': True
            }
        
        # Transformer模块处理拼接后的特征
        self.transformer = TransformerModule(
            input_dim=304,  # 256 + 48 = 304
            **transformer_config
        )
        
        # 卷积层进行最终预测
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )
        
        self._init_weight()

    def forward(self, x, low_level_feat):
        # 处理低级特征
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        # 上采样高层特征以匹配低级特征尺寸
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        # 拼接特征
        x = torch.cat((x, low_level_feat), dim=1)
        
        # 应用Transformer模块
        x = self.transformer(x)
        
        # 最终预测
        x = self.last_conv(x)
        
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, transformer_config=None):
    """
    构建解码器 - 根据配置决定是否使用Transformer
    
    Args:
        transformer_config: 如果不为None，则使用带Transformer的解码器
    """
    if transformer_config is not None:
        # 使用带Transformer的解码器
        return TransformerDecoder(num_classes, backbone, BatchNorm, transformer_config)
    else:
        # 使用标准解码器
        return Decoder(num_classes, backbone, BatchNorm)
