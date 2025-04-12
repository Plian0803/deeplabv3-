import argparse
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from subprocess import Popen, PIPE
from pathlib import Path
import threading
import queue
import re
import torch
from tqdm import tqdm
from tool.GenDataset import make_data_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import *
from tool.loss import SegmentationLosses
from tool.lr_scheduler import LR_Scheduler
from tool.saver import Saver
from tool.summaries import TensorboardSummary
from tool.metrics import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # 定义Saver和Summary用于日志记录
        self.saver = Saver(args)
        self.summary = TensorboardSummary('logs')
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)
        self.nclass = args.n_class
        self.use_self_attention = args.use_self_attention

        # 获取Transformer参数（如果使用的话）
        transformer_config = None
        if args.use_transformer:
            transformer_config = {
                'num_layers': args.num_transformer_layers,
                'num_heads': args.num_attention_heads,
                'dropout': args.dropout_rate,
                'mlp_ratio': args.mlp_ratio
            }

        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        use_self_attention=args.use_self_attention,
                        transformer_config=transformer_config)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # 创建阶段1的模型并加载预训练权重
        import importlib
        model_stage1 = getattr(importlib.import_module('network.resnet38_cls'), 'Net_CAM')(n_class=4)
        resume_stage1 = 'checkpoints/stage1_checkpoint_trained_on_' + str(args.dataset) + '.pth'
        weights_dict = torch.load(resume_stage1)
        model_stage1.load_state_dict(weights_dict)
        self.model_stage1 = model_stage1.cuda()
        self.model_stage1.eval()

        # 使用cuda并多卡并行
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        # 恢复checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                W = checkpoint['state_dict']
                if not args.ft:
                    del W['decoder.last_conv.8.weight']
                    del W['decoder.last_conv.8.bias']
                self.model.module.load_state_dict(W, strict=False)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' ".format(args.resume))
            if args.use_self_attention:
                print("=> Initializing self-attention ASPP")

        # 初始化早停计数器
        self.early_stop_counter = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target, target_a, target_b = sample['image'], sample['label'], sample['label_a'], sample['label_b']
            if self.args.cuda:
                image, target, target_a, target_b = image.cuda(), target.cuda(), target_a.cuda(), target_b.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            one = torch.ones((output.shape[0], 1, 224, 224)).cuda()
            output = torch.cat([output, (100 * one * (target == 4).unsqueeze(dim=1))], dim=1)

            loss_o = self.criterion(output, target)
            loss_a = self.criterion(output, target_a)
            loss_b = self.criterion(output, target_b)
            loss = 0.6 * loss_o + 0.2 * loss_a + 0.2 * loss_b

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## 对于label==4的像素保持原样
            pred[target == 4] = 4
            self.evaluator.add_batch(target, pred)

        # 计算验证指标
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)

        # 早停逻辑：若验证mIoU有显著提升则重置早停计数，否则增加计数器
        if mIoU > self.best_pred + self.args.min_delta:
            self.best_pred = mIoU
            self.early_stop_counter = 0
            
            # 使用模型ID创建唯一的文件名
            checkpoint_filename = 'stage2_checkpoint_'
            if self.args.model_id:
                checkpoint_filename += f"{self.args.model_id}_"
            checkpoint_filename += f"trained_on_{self.args.dataset}.pth"
            
            self.saver.save_checkpoint({
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict()
            
            }, checkpoint_filename)
        else:
            self.early_stop_counter += 1
            print("No improvement for {} epochs.".format(self.early_stop_counter))

    def load_the_best_checkpoint(self):
        # 构建相同的唯一文件名
        checkpoint_filename = 'stage2_checkpoint_'
        if self.args.model_id:
            checkpoint_filename += f"{self.args.model_id}_"
        checkpoint_filename += f"trained_on_{self.args.dataset}.pth"
        
        checkpoint_path = os.path.join(self.args.savepath, checkpoint_filename)
        print(f"加载checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)

    def test(self, epoch, Is_GM):
        self.load_the_best_checkpoint()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
                if Is_GM:
                    output = self.model(image)
                    _, y_cls = self.model_stage1.forward_cam(image)
                    y_cls = y_cls.cpu().data
                    pred_cls = (y_cls > 0.1)
            pred = output.data.cpu().numpy()
            if Is_GM:
                pred = pred * (pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy())
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## 对于label==4的像素保持原样
            pred[target == 4] = 4
            self.evaluator.add_batch(target, pred)

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Test:')
        print('[numImages: %5d]' % (i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)

def main():
    parser = argparse.ArgumentParser(description="WSSS Stage2 with Early Stopping")
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--Is_GM', type=bool, default=True, help='Enable the Gate mechanism in test phase')
    parser.add_argument('--dataroot', type=str, default='datasets/BCSS-WSSS/')
    parser.add_argument('--dataset', type=str, default='bcss')
    parser.add_argument('--savepath', type=str, default='checkpoints/')
    parser.add_argument('--workers', type=int, default=10, metavar='N')
    parser.add_argument('--sync-bn', type=bool, default=None)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])
    parser.add_argument('--n_class', type=int, default=4)
    parser.add_argument('--model_id', type=str, default='',
                   help='模型唯一标识符，用于生成不同的checkpoint文件名')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30, metavar='N')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N')
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--lr-scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
    parser.add_argument('--nesterov', action='store_true', default=False)
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    # checkpoint settings
    parser.add_argument('--resume', type=str, default='init_weights/deeplab-resnet.pth.tar')
    parser.add_argument('--checkname', type=str, default='deeplab-resnet')
    parser.add_argument('--ft', action='store_true', default=False)
    parser.add_argument('--eval-interval', type=int, default=1)
    # 功能参数
    parser.add_argument('--use_self_attention', type=bool, default=False, 
                        help='是否使用自注意力增强的ASPP')
    parser.add_argument('--use_transformer', type=bool, default=False, 
                        help='是否使用Transformer解码器')
    parser.add_argument('--num_transformer_layers', type=int, default=2, 
                        help='Transformer层数')
    parser.add_argument('--num_attention_heads', type=int, default=8, 
                        help='Transformer中的注意力头数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, 
                        help='Transformer中的Dropout率')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, 
                        help='Transformer MLP比例')
    # 早停参数
    parser.add_argument('--early_stop_patience', type=int, default=10, 
                        help='若连续多少epoch没有显著提升则提前停止训练')
    parser.add_argument('--min_delta', type=float, default=0.0, 
                        help='认为有意义的最小提升幅度')
    parser.add_argument("--pm_dir", type=str, default="",
                    help="自定义伪标签目录路径")
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    print(args)
    
    # 如果指定了自定义PM目录，覆盖默认路径
    if args.pm_dir:
        print(f"使用自定义伪标签目录: {args.pm_dir}")
        # 修改数据集路径结构
        original_train_pm = os.path.join(args.dataroot, 'train_PM')
        
        # 创建临时软链接或修改路径变量
        if os.path.exists(original_train_pm):
            # 备份原始目录(如果存在)
            backup_dir = f"{original_train_pm}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(original_train_pm, backup_dir)
            
        # 创建软链接到自定义目录
        os.symlink(args.pm_dir, original_train_pm)
        
        # 添加清理函数，确保训练结束后恢复原始目录
        def cleanup():
            os.unlink(original_train_pm)
            if os.path.exists(backup_dir):
                os.rename(backup_dir, original_train_pm)
                
        # 注册清理函数
        import atexit
        atexit.register(cleanup)
        
    trainer = Trainer(args)
    # 训练主循环，加入早停检测
    for epoch in range(trainer.args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
        # 如果早停计数器达到阈值则退出训练
        if trainer.early_stop_counter >= trainer.args.early_stop_patience:
            print("Early stopping triggered. No improvement after {} epochs.".format(trainer.args.early_stop_patience))
            break
    trainer.test(epoch, args.Is_GM)
    trainer.writer.close()

if __name__ == "__main__":
    main()
