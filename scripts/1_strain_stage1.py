import os
import numpy as np
import argparse
import importlib

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
cudnn.enabled = True

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

import matplotlib.pyplot as plt

def train_phase(args):
    model = getattr(importlib.import_module(args.network), 'Net')(args.init_gama, n_class=args.n_class)
    print(vars(args))

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])

    train_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform_train, dataset=args.dataset)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=False,
                                   drop_last=True)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights.endswith('.params'):
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    elif args.weights.endswith('.pth'):
        weights_dict = torch.load(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    else:
        print('random init')

    model = model.cuda()
    avg_meter = pyutils.AverageMeter('loss', 'avg_ep_EM', 'avg_ep_acc')
    timer = pyutils.Timer("Session started: ")

    # 👇 记录 loss 和 acc 的列表
    step_list = []
    loss_list = []
    acc_em_list = []
    acc_list = []

    for ep in range(args.max_epoches):
        model.train()
        args.ep_index = ep
        ep_count = 0
        ep_EM = 0
        ep_acc = 0

        for iter, (filename, data, label) in enumerate(train_data_loader):
            img = data
            label = label.cuda(non_blocking=True)
            enable_PDA = 1 if ep > 2 else 0
            x, feature, y = model(img.cuda(), enable_PDA)

            prob = y.cpu().data.numpy()
            gt = label.cpu().data.numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls):
                    ep_EM += 1
                acc = compute_acc(pass_cls, true_cls)
                ep_acc += acc

            avg_ep_EM = round(ep_EM / ep_count, 4)
            avg_ep_acc = round(ep_acc / ep_count, 4)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss': loss.item(),
                           'avg_ep_EM': avg_ep_EM,
                           'avg_ep_acc': avg_ep_acc})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            if optimizer.global_step % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                step_list.append(optimizer.global_step)
                loss_list.append(avg_meter.pop('loss'))
                acc_em_list.append(avg_meter.pop('avg_ep_EM'))
                acc_list.append(avg_meter.pop('avg_ep_acc'))

                print('Epoch:%2d' % (ep),
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'Loss:%.4f' % (loss_list[-1]),
                      'avg_ep_EM:%.4f' % (acc_em_list[-1]),
                      'avg_ep_acc:%.4f' % (acc_list[-1]),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'Fin:%s' % (timer.str_est_finish()),
                      flush=True)

        if model.gama > 0.65:
            model.gama *= 0.98
        print('Gama of progressive dropout attention is: ', model.gama)

    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_' + args.dataset + '.pth'))

    # 👇 保存曲线图像
    plt.figure()
    plt.plot(step_list, loss_list, label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')

    plt.figure()
    plt.plot(step_list, acc_em_list, label='Exact Match Accuracy')
    plt.plot(step_list, acc_list, label='Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.savefig('accuracy_curve.png')


def test_phase(args):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    model = model.cuda()
    args.weights = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth')
    weights_dict = torch.load(args.weights)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    score = infer(model, args.testroot, args.n_class)
    print(score)
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--session_name", default="Stage 1", type=str)
    parser.add_argument("--env_name", default="PDA", type=str)
    parser.add_argument("--model_name", default='PDA', type=str)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='datasets/BCSS-WSSS/train/', type=str)
    parser.add_argument("--testroot", default='datasets/BCSS-WSSS/test/', type=str)
    parser.add_argument("--save_folder", default='checkpoints/',  type=str)
    parser.add_argument("--init_gama", default=1, type=float)
    parser.add_argument("--dataset", default='bcss', type=str)
    args = parser.parse_args()

    #train_phase(args)
    test_phase(args)
