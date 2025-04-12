import imp
from pdb import set_trace
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms
from tool.gradcam import GradCam
def CVImageToPIL(img):
    img = img[:,:,::-1]
    img = Image.fromarray(np.uint8(img))
    return img

def PILImageToCV(img):
    img = np.asarray(img)
    img = img[:,:,::-1]
    return img

def fuse_mask_and_img(mask, img):
    mask = PILImageToCV(mask)
    img = PILImageToCV(img)
    Combine = cv2.addWeighted(mask,0.3,img,0.7,0)
    return Combine

def infer(model, dataroot, n_class):
    model.eval()
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    cam_list = []
    gt_list = []    
    bg_list = []
    transform = transforms.Compose([transforms.ToTensor()]) 
    
    print('dataroot:', dataroot)
    img_folder = os.path.join(dataroot,'img')
    print('img_folder:', img_folder)
    
    infer_dataset = Stage1_InferDataset(data_path=img_folder, transform=transform)
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]; 

        img_path = os.path.join(img_folder, os.path.basename(img_name) + '.png')
        print('img_path:', img_path)
        
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img, thr=0.25):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    cam, y = model_replicas[i%n_gpus].forward_cam(img.cuda())
                    y = y.cpu().detach().numpy().tolist()[0]
                    label = torch.tensor([1.0 if j >thr else 0.0 for j in y])
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                    return cam, label

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam-np.min(sum_cam)) / (np.max(sum_cam)-np.min(sum_cam))

        # cam --> segmap
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None)
        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        if iter%100==0:
            print(iter)
        cam_list.append(seg_map)
        gt_map_path = os.path.join(os.path.join(dataroot,'mask'), os.path.basename(img_name) + '.png')
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)
    return iouutils.scores(gt_list, cam_list, n_class=n_class)

# 添加PSM模块类
class PSMGenerator:
    def __init__(self):
        pass
        
    def generate_psm(self, feature_maps, gradients=None):
        """
        生成先验自激活映射
        feature_maps: 特征图
        gradients: 如果提供，则使用梯度信息；否则使用特征图本身
        """
        # 确保输入是tensor
        if isinstance(feature_maps, np.ndarray):
            feature_maps = torch.from_numpy(feature_maps)
            
        if gradients is None:
            # 如果没有梯度，直接使用特征图均值
            activation_maps = torch.mean(feature_maps, dim=1, keepdim=True)
        else:
            if isinstance(gradients, np.ndarray):
                gradients = torch.from_numpy(gradients)
                
            # 计算基于梯度的权重
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            # 加权特征
            activation_maps = torch.sum(feature_maps * weights, dim=1, keepdim=True)
        
        # 归一化并应用ReLU确保非负
        activation_maps = F.relu(activation_maps)
        
        return activation_maps.squeeze().cpu().numpy()
    
# 添加语义聚类模块
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
        处理激活图和原图生成语义伪标签 （二分类）
        
        参数:
            activation_map: 激活图，形状为 [H, W]
            raw_image: 原始图像，形状为 [H, W, C]
        
        返回:
            pseudo_mask: 伪标签掩码，形状为 [H, W]
        """
        # 确保数据类型正确
        activation_map = activation_map.astype(np.float32)
        
        # 处理原始图像，如果是多通道，取均值
        if len(raw_image.shape) == 3:
            raw_image_gray = np.mean(raw_image, axis=2).astype(np.float32)
        else:
            raw_image_gray = raw_image.astype(np.float32)
        
        # 归一化
        raw_image_gray = (raw_image_gray - np.min(raw_image_gray)) / (np.max(raw_image_gray) - np.min(raw_image_gray))
        
        # 融合原始信息和激活图
        fused_map = activation_map + self.beta * raw_image_gray
        
        # 将特征展平用于聚类
        h, w = fused_map.shape
        features = fused_map.reshape(-1, 1)  # 每个像素作为一个特征
        
        # 使用K-Means聚类
        from sklearn.cluster import KMeans
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
    def process_multi_class(self, activation_maps, raw_image, class_weights=None):
        """
        处理多个类别的激活图
        
        参数:
            activation_maps: 多个类别的激活图，形状为 [n_class, H, W]
            raw_image: 原始图像，形状为 [H, W, C]
            class_weights: 各类别的权重，可选
        
        返回:
            multi_class_mask: 多类别伪标签掩码，形状为 [H, W]
        """
        n_class = activation_maps.shape[0]
        h, w = activation_maps.shape[1:]
        
        # 初始化多类别掩码和置信度图
        multi_class_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)
        # 为每个类别独立生成二分类掩码
        for cls_idx in range(n_class):
            # 获取当前类别的激活图
            act_map = activation_maps[cls_idx]
            
            # 使用二分类处理
            cls_mask = self.process(act_map, raw_image)
            
            # 计算当前类别的置信度 (使用激活图的均值)
            cls_conf = np.mean(act_map[cls_mask == 1]) if np.any(cls_mask == 1) else 0
            
            # 如果有类别权重，应用它们
            if class_weights is not None:
                cls_conf *= class_weights[cls_idx]
            
            # 更新多类别掩码，保留置信度最高的类别
            update_mask = (cls_mask == 1) & ((multi_class_mask == 0) | (cls_conf > confidence_map))
            multi_class_mask[update_mask] = cls_idx + 1  # 类别标签从1开始
            confidence_map[update_mask] = cls_conf
        
        return multi_class_mask
        
        
        
        
        
        
        
        
        
        
             
def create_pseudo_mask(model, dataroot, fm, savepath, n_class, palette, dataset, use_psm=False):
    """
    生成伪标签
    
    参数:
        model: 模型
        dataroot: 数据根目录
        fm: 特征图层名称
        savepath: 保存路径
        n_class: 类别数量
        palette: 调色板
        dataset: 数据集名称
        use_psm: 是否使用PSM方法，默认为False使用原始GradCAM方法
    """
    # 原始GradCAM方法
    if fm=='b4_3':
        ffmm = model.b4_3
    elif fm=='b4_5':
        ffmm = model.b4_5
    elif fm=='b5_2':
        ffmm = model.b5_2
    elif fm=='b6':
        ffmm = model.b6
    elif fm=='bn7':
        ffmm = model.bn7
    else:
        print('error')
        return
    print(f"使用 {fm} 特征生成伪标签，数据集: {dataset}")
    
    
    # 初始化PSM生成器和语义聚类模块
    psm_generator = PSMGenerator()
    # 为不同数据集设置不同的beta值
    if dataset == 'luad':
        semantic_clustering = SemanticClusteringModule(beta=4.0, k_clusters=3)
    elif dataset == 'bcss':
        semantic_clustering = SemanticClusteringModule(beta=2.5, k_clusters=3)
    else:
        semantic_clustering = SemanticClusteringModule(beta=2.5, k_clusters=3)
    
    transform = transforms.Compose([transforms.ToTensor()])
     

    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'train'),transform=transform)

    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=False)
    
    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        print(f"当前处理图片：{img_name[0]}")      
        img_name = img_name[0]
        img_path = os.path.join(os.path.join(dataroot,'train'), os.path.basename(img_name) + '.png')
        orig_img = np.asarray(Image.open(img_path))
        
        if use_psm:
            # PSM方法: 先获取特征和梯度，然后生成PSM
            features = {}
            gradients = {}
                 
            # 注册钩子函数
            def save_features(name):
                def hook(module, input, output):
                    features[name] = output.detach()
                return hook
                
            def save_gradients(name):
                def hook(module, grad_in, grad_out):
                    gradients[name] = grad_out[0].detach()
                return hook
            
            # 根据指定的层附加钩子
            if fm == 'b4_5':
                model.b4_5.register_forward_hook(save_features('b4_5'))
                model.b4_5.register_backward_hook(save_gradients('b4_5'))
            elif fm == 'b5_2':
                model.b5_2.register_forward_hook(save_features('b5_2'))
                model.b5_2.register_backward_hook(save_gradients('b5_2'))
            elif fm == 'bn7':
                model.bn7.register_forward_hook(save_features('bn7'))
                model.bn7.register_backward_hook(save_gradients('bn7'))
                
            # 前向传播并获取类激活图
            model.eval()
            img_tensor = img_list.cuda()
            cam, y_cls = model.forward_cam(img_tensor)
            
            # 对所有类别求和作为伪损失，反向传播获取梯度
            cam_sum = torch.sum(cam)
            cam_sum.backward()
            
            # 从钩子获取特征和梯度
            feature = features[fm]
            gradient = gradients[fm]
            
            # 生成PSM
            psm_list = []
            for i in range(n_class):
                # 使用当前类别的梯度生成PSM
                gradient_for_class = gradient[:, i:i+1]
                psm = psm_generator.generate_psm(feature, gradient_for_class)
                psm_list.append(psm)
                
            # 提取标签信息
            label_str = img_name.split(']')[0].split('[')[-1]
            if dataset == 'luad':
                label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
            elif dataset == 'bcss':
                label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])
            
             # 将PSM列表转换为numpy数组
            psm_array = np.array(psm_list)
            # 使用语义聚类生成多类别伪标签
            pseudo_masks = semantic_clustering.process_multi_class(psm_array, orig_img)
            # 为LUAD数据集添加背景处理
            if dataset == 'luad':
                # 使用背景检测识别白色区域
                bg_mask = infer_utils.gen_bg_mask(orig_img)
                pseudo_masks[bg_mask > 0] = n_class + 1  # 设置为背景类
            
            visualimg = Image.fromarray(pseudo_masks.astype(np.uint8), "P")
            
        else:    
            grad_cam = GradCam(model=model, feature_module=ffmm, \
                    target_layer_names=["1"], use_cuda=True)
            cam = []
            for i in range(n_class):
                target_category = i
                grayscale_cam, _ = grad_cam(img_list, target_category)
                cam.append(grayscale_cam)
            norm_cam = np.array(cam)
            _range = np.max(norm_cam) - np.min(norm_cam)
            norm_cam = (norm_cam - np.min(norm_cam))/_range
            ##  Extract the image-level label from the filename
            ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
            ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
            label_str = img_name.split(']')[0].split('[')[-1]
            if dataset == 'luad':
                label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
            elif dataset == 'bcss':
                label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])

            cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
            cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None) #此处加入了背景，做修改
            ##  "bg_score" is the white area generated by "cv2.threshold".
            ##  Since lungs are the main organ of the respiratory system. There are a lot of alveoli (some air sacs) serving for exchanging the oxygen and carbon dioxide, which forms some white background in WSIs.
            ##  For LUAD-HistoSeg, we uses it in the pseudo-annotation generation phase to avoid some meaningless areas to participate in the training phase of stage2.
            if dataset == 'luad':
                bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
            ##  Since the white background of images of breast cancer is meaningful (e.g. fat, etc), we do not use it for the training set of BCSS-WSSS.
            elif dataset == 'bcss':
                bg_score = np.zeros((1,224,224))
                bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
            seg_map = infer_utils.cam_npy_to_label_map(bgcam_score) 
            visualimg  = Image.fromarray(seg_map.astype(np.uint8), "P")
            
        visualimg.putpalette(palette)
        try:
            visualimg.save(os.path.join(savepath, os.path.basename(img_name)+'.png'), format='PNG')
            print("成功保存 mask 文件到:", os.path.join(savepath, os.path.basename(img_name)+'.png'))
        except Exception as e:
            print("保存 mask 文件失败:", e)

        #visualimg.save(os.path.join(savepath, img_name+'.png'), format='PNG')

        if iter%100==0:           
            print(iter)
