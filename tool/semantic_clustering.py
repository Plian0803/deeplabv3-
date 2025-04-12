import numpy as np
import torch
from sklearn.cluster import KMeans

class SemanticClusteringModule:
    def __init__(self, beta=2.5, k_clusters=3):
        """
        初始化语义聚类模块
        
        参数:
            beta: 原始图像信息的权重
            k_clusters: K-Means聚类的簇数量
        """
        self.beta = beta
        self.k_clusters = k_clusters
    
    def process(self, activation_map, raw_image):
        """
        处理激活图和原图生成语义伪标签
        
        参数:
            activation_map: 激活图，形状为 [H, W]
            raw_image: 原始图像，形状为 [C, H, W]
        
        返回:
            pseudo_mask: 伪标签掩码，形状为 [H, W]
        """
        # 确保激活图是numpy格式
        if isinstance(activation_map, torch.Tensor):
            activation_map = activation_map.detach().cpu().numpy()
            
        if isinstance(raw_image, torch.Tensor):
            raw_image = raw_image.detach().cpu().numpy()
        
        # 处理原始图像，如果是多通道，取均值
        if len(raw_image.shape) == 3:
            raw_image = np.mean(raw_image, axis=0)
        
        # 融合原始信息和激活图
        fused_map = activation_map + self.beta * raw_image
        
        # 将特征展平用于聚类
        h, w = fused_map.shape
        features = fused_map.reshape(-1, 1)  # 每个像素作为一个特征
        
        # 使用K-Means聚类
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(features)
        cluster_labels = labels.reshape(h, w)
        
        # 生成伪标签 (假设强度最高的簇是前景)
        cluster_means = [np.mean(activation_map.reshape(-1)[labels == i]) for i in range(self.k_clusters)]
        foreground_cluster = np.argmax(cluster_means)
        
        # 生成二值掩码 (1为前景，0为背景)
        pseudo_mask = np.zeros_like(cluster_labels)
        pseudo_mask[cluster_labels == foreground_cluster] = 1
        
        return pseudo_mask