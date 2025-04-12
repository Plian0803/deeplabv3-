# network/psm/generate_psm.py
"""
生成 PSM‑或 Grad‑CAM‑伪标签并写入指定目录
供其他脚本 import 调用，也可作为独立 CLI 使用
"""
import os, argparse, importlib, torch
from torch.backends import cudnn
from pathlib import Path
cudnn.enabled = True

# ---- 把真正干活的函数从 infer_fun.py / 2_generate_PM.py 复制进来 ----


# -----------------------------------------------------------------------
def build_palette(dataset):
    if dataset == "luad":
        p = [0]*15
        p[0:3]  = [205, 51, 51]
        p[3:6]  = [0, 255, 0]
        p[6:9]  = [65, 105, 225]
        p[9:12] = [255, 165, 0]
        p[12:15]= [255, 255, 255]
    else:  # bcss
        p = [0]*15
        p[0:3]  = [255, 0, 0]
        p[3:6]  = [0, 255, 0]
        p[6:9]  = [0, 0, 255]
        p[9:12] = [153, 0, 255]
        p[12:15]= [255, 255, 255]
    return p


def run_generate(args):
    # 1) palette & 输出目录
    palette = build_palette(args.dataset)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 2) 加载 Stage‑1 CAM 模型
    NetCAM = getattr(importlib.import_module("network.resnet38_cls"), "Net_CAM")
    model  = NetCAM(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights), strict=False)
    model.eval().cuda()
    
    from tool.infer_fun import create_pseudo_mask   # 里面已含 PSMGenerator / SemanticClusteringModule

    # 3) 依次对 b4_5 / b5_2 / bn7 生成伪标签
    for fm in ["b4_5", "b5_2", "bn7"]:
        save_dir = out_root / f"PM_{fm}{'_PSM' if args.use_psm else ''}"
        save_dir.mkdir(exist_ok=True)
        create_pseudo_mask(model,
                           dataroot=args.dataroot,
                           fm=fm,
                           savepath=str(save_dir),
                           n_class=args.n_class,
                           palette=palette,
                           dataset=args.dataset,
                           use_psm=args.use_psm)

# -----------------------------------------------------------------------
def parse_cli():
    ap = argparse.ArgumentParser("Generate pseudo masks (PSM or Grad‑CAM)")
    ap.add_argument("--weights", default="checkpoints/stage1_checkpoint_trained_on_bcss.pth")
    ap.add_argument("--dataroot", default="datasets/BCSS-WSSS/")
    ap.add_argument("--dataset",  default="bcss", choices=["bcss", "luad"])
    ap.add_argument("--n_class",  type=int, default=4)
    ap.add_argument("--use_psm",  action="store_true")
    ap.add_argument("--output_dir", default="datasets/BCSS-WSSS/train_PM")
    return ap.parse_args()

if __name__ == "__main__":
    run_generate(parse_cli())
