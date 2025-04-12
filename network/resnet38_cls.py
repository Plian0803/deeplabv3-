import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool import infer_utils
import network.resnet38d
import torch.nn.functional as F
from tool.ADL_module import Attention_Module
from network.psm import PSMModule

      
# 修改Net类，添加自监督学习和PSM生成
class Net(network.resnet38d.Net):
    def __init__(self, gama, n_class):
        super().__init__()
        
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.Attention_Module = Attention_Module()
        self.gama = gama

        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)

        # 添加PSM模块
        self.psm_module = PSMModule()
        # 添加用于自监督学习的投影头
        self.projection_head = nn.Sequential(
            nn.Conv2d(4096, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 1, bias=False)
        )
        
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]

    # 添加自监督对比学习的前向传播
    def forward_ssl(self, x1, x2):
        # 两张增强后的图片通过编码器
        feat1 = super().forward(x1)
        feat2 = super().forward(x2)
        
        # 通过投影头
        z1 = self.projection_head(feat1)
        z2 = self.projection_head(feat2)
        
        # 池化到向量
        z1 = F.adaptive_avg_pool2d(z1, 1).view(z1.size(0), -1)
        z2 = F.adaptive_avg_pool2d(z2, 1).view(z2.size(0), -1)
        
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    #添加生成PSM的方法
    def generate_psm(self, x, target_layer='b4_5'):
        # 注册钩子来获取特征和梯度
        features = {}
        gradients = {}
        
        def save_feature_hook(name):
            def hook(module, input, output):
                features[name] = output
            return hook
            
        def save_gradient_hook(name):
            def hook(module, grad_in, grad_out):
                gradients[name] = grad_out[0]
            return hook
        
        # 根据目标层附加钩子
        if target_layer == 'b4_5':
            self.b4_5.register_forward_hook(save_feature_hook('b4_5'))
            self.b4_5.register_backward_hook(save_gradient_hook('b4_5'))
        elif target_layer == 'b5_2':
            self.b5_2.register_forward_hook(save_feature_hook('b5_2'))
            self.b5_2.register_backward_hook(save_gradient_hook('b5_2'))
        elif target_layer == 'bn7':
            self.bn7.register_forward_hook(save_feature_hook('bn7'))
            self.bn7.register_backward_hook(save_gradient_hook('bn7'))
        
        # 前向传播
        logits = self.forward_cam(x)
        
        # 对所有类别求和作为损失，进行反向传播获取梯度
        loss = torch.sum(logits)
        loss.backward()
        
        # 使用PSM模块生成激活图
        psm = self.psm_module(features[target_layer], gradients[target_layer])
        
        return psm
    
    def forward(self, x, enable_PDA):

        x = super().forward(x)  #[8,4096,28,28]
        gama = self.gama
        if enable_PDA:
            x = self.Attention_Module(x, self.fc8.weight, gama)    #[8,4096,28,28]
        else:
            x = x
        x = self.dropout7(x)    #[8,4096,28,28]
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)  #[8,4096,1,1]

        feature = x
        feature = feature.view(feature.size(0), -1)#[8,4096]


        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)
        
        return x, feature, y

    def forward_cam(self, x):
        x = super().forward(x)
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Net_CAM(network.resnet38d.Net):
    def __init__(self, n_class):
        super().__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]


    def forward(self, x):
        x = super().forward(x)
        x = self.dropout7(x)
        x = self.pool(x)
        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)
        return y

    def forward_cam(self, x):
        x_ = super().forward(x)
        x_pool = F.avg_pool2d(x_, kernel_size=(x_.size(2), x_.size(3)), padding=0)
        x = F.conv2d(x_, self.fc8.weight)
        cam = F.relu(x)
        y = self.fc8(x_pool)
        y = y.view(y.size(0), -1)
        y = torch.sigmoid(y)
        
        return cam, y


    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
