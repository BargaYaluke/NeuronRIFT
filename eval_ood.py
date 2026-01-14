# eval_ood.py
# OOD / Corruption 测试脚本（适配 Neuron-RiFT 的 checkpoint）

import argparse
import torch

from utils import *
from dataloader import *
from model import create_model


def load_checkpoint_into_model(model, ckpt_path, device="cuda"):
    """
    兼容两种格式：
      1) 直接是 state_dict
      2) {'model': state_dict, 'acc': ..., 'epoch': ...}
    """
    print(f"==> Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print("[Warning] Missing keys:", missing)
    if len(unexpected) > 0:
        print("[Warning] Unexpected keys:", unexpected)

    return model


def main():
    parser = argparse.ArgumentParser(description='OOD / corruption evaluation (Neuron-RiFT)')

    # ---- 和训练脚本保持统一的参数风格 ----
    parser.add_argument('--dataset', default="CIFAR100", type=str,
                        choices=["CIFAR10", "CIFAR100", "TinyImageNet"],
                        help="ID 训练时用的数据集（决定 num_classes / input_size 等）")
    parser.add_argument('--model', default="ResNet18", type=str, help='model used')

    # 允许自动推 num_classes / input_size，也可以手动指定覆盖
    parser.add_argument('--num_classes', default=None, type=int, help='num classes')
    parser.add_argument('--input_size', default=None, type=int, help='input_size')
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')

    parser.add_argument('--device', default="cuda", type=str, help='device')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')

    # 要评估的 checkpoint 路径（必须给）
    parser.add_argument('--resume', required=True, type=str,
                        help='path to checkpoint to be evaluated (e.g. best_params.pth)')

    args = parser.parse_args()

    # ---- 根据 dataset 自动补 num_classes / input_size ----
    if args.num_classes is None:
        if args.dataset == "CIFAR10":
            args.num_classes = 10
        elif args.dataset == "CIFAR100":
            args.num_classes = 100
        elif args.dataset == "TinyImageNet":
            args.num_classes = 200   # Tiny-ImageNet 有 200 类
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.input_size is None:
        if "CIFAR" in args.dataset:
            args.input_size = 32
        else:
            args.input_size = 64   # Tiny-ImageNet 通常是 64x64

    print("==> Args for OOD eval:")
    print(args)

    # ---- 1. 创建模型骨架（这里不从 create_model 里加载权重）----
    model = create_model(
        args.model,
        args.input_size,
        args.num_classes,
        args.device,
        args.patch,
        resume=None  # 我们自己手动 load
    )

    # ---- 2. 手动加载 checkpoint（兼容 state_dict / {'model': ...}）----
    model = load_checkpoint_into_model(model, args.resume, args.device)

    # ---- 3. 调用 RIFT 原来的 corruption 评估接口 ----
    if args.dataset == "CIFAR10":
        corruption_acc_dict = evaluate_cifar_corruption(
            args, model, data_dir="/data/coding/RiFT/CIFAR-10-C"
        )
    elif args.dataset == "CIFAR100":
        corruption_acc_dict = evaluate_cifar_corruption(
            args, model, data_dir="./data/CIFAR-100-C"
        )
    elif args.dataset == "TinyImageNet":
        corruption_acc_dict = evaluate_tiny_corruption(args, model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 打印每种 corruption 的精度 + 平均
    print("\n==> Corruption accuracy dict:")
    for k, v in corruption_acc_dict.items():
        print(f"{k:20s}: {v:.2f}%")

    if "mean_acc" in corruption_acc_dict:
        print(f"\n==> Mean corruption acc: {corruption_acc_dict['mean_acc']:.2f}%")

if __name__ == "__main__":
    main()
