import os
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
from tool.infer_fun import create_pseudo_mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='checkpoints/stage1_checkpoint_trained_on_bcss.pth', type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--dataroot", default="datasets/BCSS-WSSS/", type=str)
    parser.add_argument("--dataset", default="bcss", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--use_psm", action="store_true", help="使用PSM方法生成伪标签，否则使用原始的GradCAM方法")
    parser.add_argument("--unique_output_dir", type=str, default="",
                    help="为此次运行设置唯一输出目录")
    args = parser.parse_args()
    print(args)
    if args.dataset == 'luad':
        palette = [0]*15
        palette[0:3] = [205,51,51]
        palette[3:6] = [0,255,0]
        palette[6:9] = [65,105,225]
        palette[9:12] = [255,165,0]
        palette[12:15] = [255, 255, 255]
    elif args.dataset == 'bcss':
        palette = [0]*15
        palette[0:3] = [255, 0, 0]
        palette[3:6] = [0,255,0]
        palette[6:9] = [0,0,255]
        palette[9:12] = [153, 0, 255]
        palette[12:15] = [255, 255, 255]
        
    # 修改输出目录的设置逻辑
    if args.unique_output_dir:
        # 使用传入的唯一目录
        PMpath = args.unique_output_dir
        # 确保目录存在，包括父目录
        os.makedirs(PMpath, exist_ok=True)
    else:
        # 使用默认目录
        PMpath = os.path.join(args.dataroot, 'train_PM')
        if not os.path.exists(PMpath):
            os.mkdir(PMpath)
            
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights), strict=False)
    model.eval()
    model.cuda()
    
    # 在目录名中添加方法标识
    method_tag = "_PSM" if args.use_psm else ""
    
    
    
    ##
    fm = 'b4_5'
    savepath = os.path.join(PMpath, f'PM_{fm}{method_tag}')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset, args.use_psm)
    
    ##
    fm = 'b5_2'
    savepath = os.path.join(PMpath, f'PM_{fm}{method_tag}')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset, args.use_psm)
    
    ##
    fm = 'bn7'
    savepath = os.path.join(PMpath, f'PM_{fm}{method_tag}')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset, args.use_psm)
