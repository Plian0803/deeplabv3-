import torch
import torch.nn as nn
import torch.nn.functional as F


# 添加PSM模块类

class PSMModule(nn.Module):
    def __init__(self):
        super(PSMModule, self).__init__()
        
    def forward(self, feature_maps, gradients=None):
        """
        生成先验自激活映射
        feature_maps: 特征图
        gradients: 如果提供，则使用梯度信息；否则使用特征图本身
        """
        if gradients is None:
            # 如果没有梯度，直接使用特征图均值
            activation_maps = torch.mean(feature_maps, dim=1, keepdim=True)
        else:
            # 计算基于梯度的权重
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            # 加权特征
            activation_maps = torch.sum(feature_maps * weights, dim=1, keepdim=True)
        
        # 归一化并应用ReLU确保非负
        activation_maps = F.relu(activation_maps)
        
        return activation_maps