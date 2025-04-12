import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from network.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from network.aspp import build_aspp, build_self_attention_aspp  # 引入
from network.decoder import build_decoder
from network.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, use_self_attention=False, transformer_config=None):
        """
        DeepLab V3+ 模型
        
        Args:
            backbone: 骨干网络类型
            output_stride: 输出步长
            num_classes: 类别数量
            sync_bn: 是否使用同步批归一化
            freeze_bn: 是否冻结批归一化层
            use_self_attention: 是否使用自注意力增强的ASPP
            transformer_config: Transformer配置，如果为None则不使用Transformer
        """
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # 保存模块的使用配置，便于后续参数分组
        self.use_self_attention = use_self_attention
        self.use_transformer = transformer_config is not None
        
         # 根据参数选择ASPP版本
        if use_self_attention:
            self.aspp = build_self_attention_aspp(backbone, output_stride, BatchNorm)
        else:
            self.aspp = build_aspp(backbone, output_stride, BatchNorm)
            
         # 使用通用的解码器构建函数，传入transformer_config参数    
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, transformer_config = transformer_config)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        # print(x.shape)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        """获取骨干网络参数（使用较小学习率）"""
        params = []
        # 记录已添加的参数ID以避免重复
        added_params = set()
        
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad and id(p) not in added_params:
                                params.append(p)
                                added_params.add(id(p))
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad and id(p) not in added_params:
                                params.append(p)
                                added_params.add(id(p))
        return params

    def get_10x_lr_params(self):
        """获取ASPP和解码器参数（使用较大学习率）"""
        params = []
        added_params = set()
        
        # 基本模块总是包含在10x参数组
        modules = [self.aspp, self.decoder]
        
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                # 检查常规模块
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad and id(p) not in added_params:
                                params.append(p)
                                added_params.add(id(p))
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad and id(p) not in added_params:
                                params.append(p)
                                added_params.add(id(p))
                
                # 额外检查Transformer相关模块
                if self.use_transformer:
                    # 检查Transformer特有的层
                    if isinstance(m[1], nn.Linear) or isinstance(m[1], nn.LayerNorm) or \
                    'TransformerModule' in str(type(m[1])) or 'TransformerBlock' in str(type(m[1])) or \
                    'EfficientAttention' in str(type(m[1])):
                        for p in m[1].parameters():
                            if p.requires_grad and id(p) not in added_params:
                                params.append(p)
                                added_params.add(id(p))
                
                # 检查自注意力模块
                if self.use_self_attention:
                    if 'SelfAttentionModule' in str(type(m[1])):
                        for p in m[1].parameters():
                            if p.requires_grad and id(p) not in added_params:
                                params.append(p)
                                added_params.add(id(p))
        
        return params
#%%
if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
    summary(model.cuda(),(3,513,513))
    
    

