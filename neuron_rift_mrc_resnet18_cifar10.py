import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import re
import torchattacks  # 需要提前 pip install torchattacks
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
from tqdm import tqdm
from loss_utils import SupConLoss
import json
from datetime import datetime

from utils import *
from dataloader import *
from model import create_model
from optimizer import *
from torchvision import datasets

from robustbench.utils import load_model


def generate_adv_dataset(args, model):
    adv_train_dataset = adv_dataset()
    model = model.eval()

    # --- mean/std 与训练一致 ---
    if "CIFAR" in args.dataset:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=args.device).view(1,3,1,1)
        std  = torch.tensor([0.2471, 0.2435, 0.2616], device=args.device).view(1,3,1,1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1,3,1,1)

    # 1) 数据集：只 ToTensor，保证输入范围 [0,1]
    transform_plain = transforms.Compose([transforms.ToTensor()])

    use_train_split = (getattr(args, "adv_source", "train") == "train")
    if args.dataset == "CIFAR10":
        base_dataset = datasets.CIFAR10(root="./data", train=use_train_split,
                                        download=True, transform=transform_plain)
    elif args.dataset == "CIFAR100":
        base_dataset = datasets.CIFAR100(root="./data", train=use_train_split,
                                         download=True, transform=transform_plain)
    else:
        split = "train" if use_train_split else "val"
        base_dataset = TinyImageNet(split, transform_plain)

    # 2) 固定子集采样
    subset_size = int(getattr(args, "adv_subset", 10000))
    subset_size = min(subset_size, len(base_dataset))
    g = torch.Generator()
    g.manual_seed(int(getattr(args, "adv_seed", 0)))
    perm = torch.randperm(len(base_dataset), generator=g)[:subset_size].tolist()
    sampler = SubsetRandomSampler(perm)

    base_loader = torch.utils.data.DataLoader(
        base_dataset, batch_size=128, sampler=sampler, shuffle=False,
        num_workers=8, pin_memory=True
    )

    # 3) 攻击：在像素空间 [0,1] 做 PGD（torchattacks 的默认假设）
    atk_model = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)

    # 4) 生成并存储：adv 先 clamp 到 [0,1]，再 normalize 后存入 adv_dataset
    for images, labels in base_loader:
        images = images.to(args.device, non_blocking=True)    # [0,1]
        labels = labels.to(args.device, non_blocking=True)

        adv_images = atk_model(images, labels)                # 仍应在 [0,1]
        adv_images = adv_images.clamp(0.0, 1.0)

        adv_norm = (adv_images - mean) / std                  # normalize 后用于模型
        adv_train_dataset.append_data(adv_norm.detach(), labels.detach())

    return adv_train_dataset



def neuron_mrc_and_prune(args, model,
                         num_grad_batches=10,
                         neurons_per_layer=1,
                         epsilon=0.1):

    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    # 对比学习损失（单视图即可）
    supcon = SupConLoss(temperature=getattr(args, "temperature", 0.07))

    # 混合损失系数（可在命令行加 --lambda_con=0.1 调参；不加就用默认）
    lambda_ce  = getattr(args, "lambda_ce", 1.0)
    lambda_con = getattr(args, "lambda_con", 0.1)

    # 是否启用对比项（默认启用；如需只用 CE，可传 --contrastive=False）
    use_contrastive = getattr(args, "contrastive", False)


    # ------------------------- 2. 生成对抗样本数据集 -------------------------
    adv_dataset = generate_adv_dataset(args, deepcopy(model))
    adv_loader = torch.utils.data.DataLoader(
        adv_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0
    )

    # ------------------------- 3. 先计算当前鲁棒 Loss / Acc （和原 layer_sharpness 一样） -------------------------
    origin_total = 0
    origin_loss = 0.0
    origin_correct = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in adv_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            bs = targets.shape[0]
            origin_total += bs
            origin_loss += loss.item() * bs
            _, predicted = outputs.max(1)
            origin_correct += predicted.eq(targets).sum().item()

    origin_acc = origin_correct / origin_total
    origin_loss = origin_loss / origin_total

    args.logger.info(
        "{:35}, Robust Loss: {:10.4f}, Robust Acc: {:10.2f}".format(
            "Origin", origin_loss, origin_acc * 100.0
        )
    )


    def _two_views_tensor(x, pad=4):
        """
        x: [N,C,H,W] 的 tensor（你这里是已经 Normalize 过的 adv_norm）
        返回两份随机增强 view：v1, v2
        增强只做：随机水平翻转 + 随机裁剪（带 padding）
        """
        def _aug(z):
            # random horizontal flip
            if torch.rand(1, device=z.device).item() < 0.5:
                z = torch.flip(z, dims=[3])  # W 维

            # random crop with padding
            if pad > 0:
                z = F.pad(z, (pad, pad, pad, pad), mode="reflect")
                _, _, H, W = z.shape
                # 原始尺寸
                h0 = H - 2 * pad
                w0 = W - 2 * pad
                top = torch.randint(0, 2 * pad + 1, (1,), device=z.device).item()
                left = torch.randint(0, 2 * pad + 1, (1,), device=z.device).item()
                z = z[:, :, top:top + h0, left:left + w0]

            return z

        return _aug(x), _aug(x)

    def _extract_logits_and_feats(forward_model, x):
        """
        尝试从（norm_layer, backbone）封装后的模型拿到 backbone 的倒数第二层特征；
        若失败，则直接用 logits 作为特征。
        返回：logits [N,C]，feats [N,D]
        """
        # 1) 拿到真正的 backbone
        if isinstance(forward_model, nn.Sequential) and len(forward_model) >= 2:
            inner = forward_model[1]
        else:
            inner = forward_model

        captured = {}
        handle = None

        # 2) 在 backbone 里找到“最后一个 nn.Linear”
        last_linear = None
        for m in inner.modules():
            if isinstance(m, nn.Linear):
                last_linear = m

        if last_linear is not None:
            def _hook(mod, inp, out):
                # inp 是一个 tuple，inp[0] 形状大概是 [N, D]
                z = inp[0]
                if z.dim() > 2:
                    z = torch.flatten(z, 1)
                captured["feat"] = z
            handle = last_linear.register_forward_hook(_hook)
        else:
            # 3) 实在没有 Linear，就兜底用常见名字再试试（以防换了其它 backbone）
            for cand in ["avgpool", "global_pool", "pool", "pre_logits"]:
                if hasattr(inner, cand):
                    layer = getattr(inner, cand)
                    def _hook(mod, inp, out):
                        z = out
                        if z.dim() > 2:
                            z = torch.flatten(z, 1)
                        captured["feat"] = z
                    handle = layer.register_forward_hook(_hook)
                    break

        # 4) 正常前向，会触发上面的 hook
        logits = forward_model(x)
        if handle is not None:
            handle.remove()

        # 5) 如果 hook 成功，就用 captured["feat"]；否则退化为 logits
        if "feat" in captured:
            print('YES FEAT')
            feats = captured["feat"]
        else:
            print("NO FEAT (use logits as features)")
            feats = logits

        return logits, feats

    # ------------------------- 4. 在对抗样本上累积鲁棒损失梯度（Neuron MRC 的核心） -------------------------
    for p in model.parameters():
        p.grad = None

    model.eval()  

    # 注意：adv_loader 是可重复遍历的 DataLoader，再来一遍没问题
    for batch_idx, (inputs, targets) in enumerate(adv_loader):
        if batch_idx >= num_grad_batches:
            break
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        # logits 与 feats（优先抓倒数第二层；抓不到就用 logits）
        logits1, feat1 = _extract_logits_and_feats(model, inputs)
        ce1 = criterion(logits1, targets)

        if use_contrastive:
            # 两视图输入增强 + forward 第二次
            v1, v2 = _two_views_tensor(inputs, pad=4)

        logits1, feat1 = _extract_logits_and_feats(model, v1)

        with torch.no_grad():
            logits2, feat2 = _extract_logits_and_feats(model, v2)

        ce = criterion(logits1, targets)   # 选择阶段用一个 view 的 CE 就够
        feats_2v = torch.stack([feat1, feat2], dim=1)
        con = supcon(feats_2v, labels=targets)

        loss = (lambda_ce * ce + lambda_con * con) / num_grad_batches


        loss.backward()

    # 至此，model.named_parameters() 中的 param.grad
    # 存的是鲁棒损失（对抗样本 CE）的梯度（平均意义）
        # 在 5. 神经元级 MRC 之前，加一个梯度检查
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad
        has_nan = torch.isnan(g).any().item()
        has_inf = torch.isinf(g).any().item()
        if has_nan or has_inf:
            print(f"[GRAD CHECK] {name}: nan={has_nan}, inf={has_inf}, max_abs_grad={g.abs().max().item():.4e}")


    # ------------------------- 5. 神经元级 MRC + GPS 风格 mask -------------------------
    neuron_mrc_list = []   # [(param_name, neuron_idx, mrc)]
    statistic = {}         # {param_name: [trainable_param, total_param]}
    new_masks = {}         # {param_name: mask_tensor}
    selected_neurons_per_layer = {} 

    for name, param in model.named_parameters():
        # 5.0 没有梯度的参数，直接全冻结
        if param.grad is None:
            new_mask = np.ones_like(param.data.cpu().numpy(), dtype=np.float32)
            statistic[name] = [0, new_mask.size]
            new_masks[name] = torch.from_numpy(new_mask).to(args.device)
            continue

        # 5.1 非 weight（bias / BN.gamma 等）：
        if not name.endswith(".weight"):
            # 这里也全部冻结：mask = 1
            new_mask = np.ones_like(param.data.cpu().numpy(), dtype=np.float32)
            statistic[name] = [0, new_mask.size]
            new_masks[name] = torch.from_numpy(new_mask).to(args.device)
            continue

        # 5.2 有梯度的 weight：Conv/Linear 做神经元级处理
        grad = param.grad.data.cpu().numpy()
        weight = param.data.cpu().numpy()

        # Conv2d: [C_out, C_in, kH, kW]，每个输出通道 = 一个“神经元”
        # Linear: [out_features, in_features]，每个输出单元 = 一个“神经元”
        if grad.ndim == 4:
            B, C, H, W = grad.shape
            grad_2d = grad.reshape(B, -1)
            weight_2d = weight.reshape(B, -1)
        elif grad.ndim == 2:
            B, _ = grad.shape
            grad_2d = grad
            weight_2d = weight.reshape(B, -1)                # 每行 = 一个输出单元
        else:
            # 其它形状（比如 BN.weight 是 1 维），这里简单全冻结
            new_mask = np.ones_like(param.data.cpu().numpy(), dtype=np.float32)
            statistic[name] = [0, new_mask.size]
            new_masks[name] = torch.from_numpy(new_mask).to(args.device)
            continue

        # -------------------- 统计逻辑部分 --------------------
        grad_norm   = np.linalg.norm(grad_2d, axis=1)
        weight_norm = np.linalg.norm(weight_2d, axis=1)

        # 分类
        zero_weight = (weight_norm == 0)                       # 权重完全为0（剪枝造成）
        alive_zero_grad = (weight_norm > 0) & (grad_norm == 0) # 权重非0但梯度为0（可能ReLU死）
        alive_with_grad = (weight_norm > 0) & (grad_norm > 0)  # 正常神经元

        # 打印统计结果
        print(f"[{name}] "
            f"zero_weight={zero_weight.sum()}, "
            f"alive_zero_grad={alive_zero_grad.sum()}, "
            f"alive_with_grad={alive_with_grad.sum()}")

        # 5.3 计算每个“神经元”的相对 MRC ≈ ||grad||_2 / (||w||_2 + eps)
        eps = 1e-12
        mrc_per_neuron = grad_norm / (weight_norm + eps)

        for j in range(B):
            neuron_mrc_list.append(
                (name, int(j), float(mrc_per_neuron[j]))
            )

        # 5.4 在当前层内选 MRC 最小的 neurons_per_layer 个神经元作为“可训练”
        valid_mask = (weight_norm > 0) & (grad_norm > 0)
        # 如果这一层所有神经元都是无效的，就直接跳过选择
        if valid_mask.sum() == 0:
            selected_indices = np.array([], dtype=int)
        else:
            # 把无效神经元的 MRC 设成 +inf，保证不会被选中
            mrc_for_select = mrc_per_neuron.copy()
            mrc_for_select[~valid_mask] = np.inf

            # 在“有梯度”的神经元中选最小的 k 个
            k = min(neurons_per_layer, valid_mask.sum())
            selected_indices = np.argsort(mrc_for_select)[:k]

        # 记录当前层被选中的神经元及其 MRC
        selected_info = []
        for idx in selected_indices:
            selected_info.append((int(idx), float(mrc_per_neuron[idx])))
        selected_neurons_per_layer[name] = selected_info

        # 与 GPS 一致：mask=0 → 可训练；mask=1 → 冻结
        weight_np = param.data.cpu().numpy()
        new_mask = np.ones_like(weight_np, dtype=np.float32)

        if grad.ndim == 4:
            # Conv: 把 selected_indices 对应的整个输出通道置 0
            new_mask[selected_indices, :, :, :] = 0.0
        elif grad.ndim == 2:
            # Linear: 把对应输出单元整行置 0
            new_mask[selected_indices, :] = 0.0

        # 统计这个参数张量里“可训练”（mask=0）的元素个数
        trainable_param = new_mask.size - np.count_nonzero(new_mask)
        total_para = new_mask.size
        statistic[name] = [int(trainable_param), int(total_para)]

        print(
            name, ": ",
            trainable_param, "/", total_para,
            "(", np.round((trainable_param / total_para) * 100, 4), "%)",
            new_mask.shape
        )

        new_masks[name] = torch.from_numpy(new_mask).to(args.device)

    # ------------------------- 6. 全网络神经元按 MRC 排序 & 打印 -------------------------
    neuron_mrc_list = [t for t in neuron_mrc_list if t[2] > 0]
    neuron_mrc_list.sort(key=lambda x: x[2])

    print("==> Per-layer neuron with MINIMUM MRC (gradient norm):")

    # 为每一层找出 MRC 最小的神经元
    layer_best = {}  # {layer_name: (neuron_idx, mrc)}

    for layer_name, neuron_idx, mrc in neuron_mrc_list:
        # 如果这一层还没有记录，或者当前 mrc 更小，就更新
        if (layer_name not in layer_best) or (mrc < layer_best[layer_name][1]):
            layer_best[layer_name] = (neuron_idx, mrc)

    # 按层名排序打印（可选）
    for layer_name in sorted(layer_best.keys()):
        idx, mrc = layer_best[layer_name]
        print(f"{layer_name:40s}  min_MRC_neuron={idx:4d},  MRC≈||grad||={mrc:.6e}")
        

    # 之后 finetune 时：
    #   for name, param in model.named_parameters():
    #       if name in new_masks and param.grad is not None:
    #           param.grad.mul_(new_masks[name])
    # 就可以实现“只更新 MRC 最小的神经元”
    return neuron_mrc_list, new_masks, statistic, selected_neurons_per_layer

# -------------------------
# 5. 主函数：串起整个流程
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')

# ---------- 原有通用参数 ----------
    parser.add_argument('--model', default="ResNet18", type=str, help='model used')
    parser.add_argument('--dataset', default="CIFAR10", type=str,
                    help="dataset", choices=["CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--input_size', default=32, type=int, help='input_size')

# ---------- RiFT 层级 MRC ----------
    parser.add_argument("--cal_mrc", action="store_true",
                        help='If to calculate Module Robust Criticality (MRC) value of each module.')

# ---------- ✅ 新增：神经元级 MRC ----------
    parser.add_argument("--cal_neuron_mrc", action="store_true",
                        help='If to calculate Neuron-level Robust Criticality (Neuron-MRC) value.')

# 可选控制参数（对应 neuron_mrc_and_prune() 函数的输入）
    parser.add_argument("--neurons_per_layer", default=1, type=int,
                        help="number of least-robust neurons (per layer) to select as trainable")
    parser.add_argument("--num_grad_batches", default=10, type=int,
                        help="number of adversarial batches used to accumulate gradient for Neuron-MRC")
    parser.add_argument("--epsilon", default=0.1, type=float,
                        help="perturbation bound (for consistency with layer-level RiFT)")

# ---------- 训练参数 ----------
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--patch', default=4, type=int, help='num patch (used by vit)')
    parser.add_argument('--optim', default="SGDM", type=str, help="optimizer")
    parser.add_argument('--device', default="cuda", type=str, help='device')
    parser.add_argument('--lr_scheduler', default="step", choices=["step", 'cosine'])
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGDM')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='lr_decay_gamma')
    parser.add_argument('--wd', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--epochs', default=10, type=int, help='num of epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--contrastive', action='store_true', default=False)
    parser.add_argument('--lambda_con', type=float, default=0.1)
    parser.add_argument('--lambda_ce', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--robust_eval_interval', type=int, default=1,
                    help='how many epochs between robust evaluations')
    parser.add_argument("--adv_source", type=str, default="train", choices=["train", "test"],
                    help="which split to generate adversarial samples for Neuron-MRC measurement")
    parser.add_argument("--adv_subset", type=int, default=10000,
                    help="subset size used to generate adversarial samples for Neuron-MRC")
    parser.add_argument("--adv_seed", type=int, default=0,
                    help="random seed used to sample the adv_subset indices")
    parser.add_argument("--mask_mode", type=str, default="grad", choices=["grad", "param"],
                    help="grad: gate gradients; param: hard-freeze frozen entries after optimizer step")




    return parser.parse_args()

@torch.no_grad()
def apply_param_freeze(model, neuron_masks, init_sd, device):
    # neuron_masks: mask=0 可训练, mask=1 冻结（与你保存的定义一致）
    for name, p in model.named_parameters():
        if name not in neuron_masks:
            continue
        mask = neuron_masks[name].to(device, non_blocking=True)
        train_mask = 1.0 - mask  # 1=可训练, 0=冻结

        w0 = init_sd[name].to(device, non_blocking=True)
        # 冻结位置覆盖回 w0，可训练位置保持当前值
        p.data.mul_(train_mask).add_(w0 * (1.0 - train_mask))

def training_one_epoch(
    args,
    model,
    trainloader,
    optimizer,
    criterion,
    epoch,
    logger,
    neuron_masks=None,   # {param_name: mask_tensor}，0=可训练，1=冻结
    mixup_fn=None,       # 以后你想加 SupCon / mixup 可以用
):
    """
    单轮训练：
      - 结构上更接近 GPS 的 train_one_epoch（有 batch 日志和时间统计）
      - 保留 RiFT 的简单 CE 训练逻辑
      - 关键：支持神经元级 mask（梯度门控）
    """
    model.train()
    device = args.device

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start = time.time()
    num_batches = len(trainloader)
    log_interval = getattr(args, "log_interval", 50)  # 没设就默认 50

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # ------------------------ data to device ------------------------
        data_time = time.time() - start

        inputs = inputs.to(device)
        targets = targets.to(device)

        # ------------------------ forward & loss ------------------------
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size

        # ------------------------ backward ------------------------
        loss.backward()

        # ---- 关键：神经元级 mask（0=可训练，1=冻结） ----
        if neuron_masks is not None:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if name not in neuron_masks:
                        continue

                    mask = neuron_masks[name]
                    # 确保 mask 在同一设备
                    if mask.device != param.grad.device:
                        mask = mask.to(param.grad.device)
                        neuron_masks[name] = mask  # 顺手缓存一下

                    # 约定：mask=0 → 该位置“允许更新”；mask=1 → 冻结
                    # 因此梯度应该在 mask==1 的位置清零：
                    #  train_mask = 1 - mask : (1 表示可更新)
                    train_mask = 1.0 - mask
                    param.grad.mul_(train_mask)

        optimizer.step()

        if neuron_masks is not None and getattr(args, "mask_mode", "grad") == "param":
            apply_param_freeze(model, neuron_masks, args.init_sd, device)
        



        # ------------------------ 统计精度 ------------------------
        # 注意：如果后面你用 mixup / label-smoothing，这种 hard label 精度就不再准确了
        _, predicted = outputs.max(1)
        total_samples += batch_size
        total_correct += predicted.eq(targets).sum().item()

        # ------------------------ 日志 & 计时 ------------------------
        batch_time = time.time() - start

        if (batch_idx % log_interval == 0) or (batch_idx == num_batches - 1):
            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = 100.0 * total_correct / max(total_samples, 1)
            logger.info(
                "Epoch [{}/{}]  Batch [{}/{}]  "
                "Loss: {:.4f} (avg {:.4f})  "
                "Acc: {:.2f}% (avg {:.2f}%)  "
                "Data: {:.3f}s  Batch: {:.3f}s".format(
                    epoch, args.epochs,
                    batch_idx, num_batches,
                    loss.item(), avg_loss,
                    (100.0 * predicted.eq(targets).float().mean().item()), avg_acc,
                    data_time, batch_time,
                )
            )

        start = time.time()

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_acc = 100.0 * total_correct / max(total_samples, 1)

    logger.info(
        "==> Epoch {} done. Train loss: {:.4f}, train acc: {:.2f}%".format(
            epoch, epoch_loss, epoch_acc
        )
    )

    return epoch_loss, epoch_acc


import time

def validate_clean(args, model, loader, criterion, logger, prefix=""):
    """
    只在干净 test set 上做验证：
      - 返回 test_loss, test_acc
      - 没有鲁棒性评估
    """
    logger.info("==> {}Clean validating...".format(prefix))
    model.eval()

    device = args.device
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    start = time.time()
    num_batches = len(loader)
    log_interval = getattr(args, "log_interval", 50)  # 没设的话默认 50

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            data_time = time.time() - start

            inputs = inputs.to(device)
            targets = targets.to(device)

            # 可选：channels_last 优化（和 GPS 对齐的接口）
            if getattr(args, "channels_last", False):
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size

            _, predicted = outputs.max(1)
            total_samples += batch_size
            total_correct += predicted.eq(targets).sum().item()

            batch_time = time.time() - start

            if (batch_idx % log_interval == 0) or (batch_idx == num_batches - 1):
                avg_loss = total_loss / max(total_samples, 1)
                avg_acc = 100.0 * total_correct / max(total_samples, 1)
                batch_acc = 100.0 * predicted.eq(targets).float().mean().item()

                logger.info(
                    "{}Test(clean): [{:>4d}/{}]  "
                    "Loss: {:.4f} (avg {:.4f})  "
                    "Acc@1: {:.2f}% (avg {:.2f}%)  "
                    "Data: {:.3f}s  Batch: {:.3f}s".format(
                        prefix,
                        batch_idx, num_batches,
                        loss.item(), avg_loss,
                        batch_acc, avg_acc,
                        data_time, batch_time,
                    )
                )

            start = time.time()

    test_loss = total_loss / max(total_samples, 1)
    test_acc = 100.0 * total_correct / max(total_samples, 1)

    logger.info(
        "==> {}Clean test done. loss: {:.4f}, acc: {:.2f}%".format(
            prefix, test_loss, test_acc
        )
    )

    return test_loss, test_acc

def validate_robust(args, model, loader, criterion, eval_robustness_func, logger, prefix=""):
    """
    完整验证：
      1) 先在干净 test set 上跑一次 validate_clean
      2) 再调用 eval_robustness_func(args, model) 计算鲁棒精度
    """
    # 1. 干净测试
    test_loss, test_acc = validate_clean(
        args, model, loader, criterion, logger, prefix=prefix
    )

    # 2. 鲁棒评估（PGD / TinyImageNet-C 等）
    test_robust_acc = eval_robustness_func(args, model)

    logger.info(
        "==> {}Robust validate done. "
        "clean acc: {:.2f}%, robust acc: {:.2f}%".format(
            prefix, test_acc, test_robust_acc
        )
    )

    return test_loss, test_acc, test_robust_acc


def main():
    args = parse_args()
    
    experiment_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "dataset": args.dataset,
        "neurons_per_layer": args.neurons_per_layer,
        "epochs": args.epochs,
        "lr": args.lr,
        "results": [],
        "selection_phase": {},
        "final_result": {}
    }

    # -------------------- 基本设置 --------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    proj_name = "rift"
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    suffix = '{}_{}_lr={}_wd={}_epochs={}_neurons={}'.format(
    proj_name, args.optim, args.lr, args.wd, args.epochs, args.neurons_per_layer
    )
    # 模型 checkpoint 目录
    model_save_dir = f'./results/{args.model}_{args.dataset}/checkpoint/{suffix}/'
    os.makedirs(model_save_dir, exist_ok=True)

    # 日志路径
    logger = create_logger(os.path.join(model_save_dir, 'output.log'))
    logger.info(args)
    args.logger = logger

    # mask 文件统一管理路径（建议单独目录）
    mask_dir = f'./results/{args.model}_{args.dataset}/masks/'
    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, f'neuron_masks_neurons={args.neurons_per_layer}.pth')
    

    # -------------------- 数据 --------------------
    logger.info('==> Preparing data and create dataloaders...')
    if "CIFAR" in args.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2471, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2471, 0.2435, 0.2616)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    transform_dict = {"train": transform_train, "test": transform_test}

    trainloader, _, testloader = create_dataloader(
        args.dataset, args.batch_size, use_val=False, transform_dict=transform_dict
    )

    logger.info('==> Building dataloaders...')
    logger.info(args.dataset)

    # -------------------- 模型 & 优化器 --------------------
    logger.info('==> Building model...')
    model = create_model(
        args.model, args.input_size, args.num_classes,
        args.device, args.patch, args.resume
    )
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print("[PARAM NAN]", name)
        if torch.isinf(p).any():
            print("[PARAM INF]", name)
    logger.info(args.model)

    logger.info('==> Building optimizer and learning rate scheduler...')
    optimizer = create_optimizer(
        args.optim, model, args.lr, args.momentum, weight_decay=args.wd
    )
    logger.info(optimizer)

    lr_decays = [int(args.epochs // 2)]
    scheduler = create_scheduler(args, optimizer, lr_decays=lr_decays)
    logger.info(scheduler)

    criterion = nn.CrossEntropyLoss()

    init_sd = deepcopy(model.state_dict())
    torch.save(init_sd, model_save_dir + "init_params.pth")
    args.init_sd = init_sd

    # 选择鲁棒评估函数
    if "CIFAR" in args.dataset:
        evalulate_robustness = evaluate_cifar_robustness
    else:
        evalulate_robustness = evaluate_tiny_robustness

    # ======================================================
    # 1) 原 RiFT：模块级 MRC 模式
    # ======================================================
    if args.cal_mrc:
        layer_sharpness(args, deepcopy(model), epsilon=0.1)
        logger.info("==> Layer-wise MRC (RiFT) computed, exit.")
        return

    # ======================================================
    # 2) 新增：神经元级 MRC 模式（Neuron-RiFT）
    # ======================================================
    if getattr(args, "cal_neuron_mrc", False):
        neuron_mrc_list, new_masks, statistic, selected_neurons_per_layer = neuron_mrc_and_prune(
            args,
            deepcopy(model),
            num_grad_batches=args.num_grad_batches,
            neurons_per_layer=args.neurons_per_layer,
            epsilon=args.epsilon,
        )

        # 保存神经元级 MRC 结果
        np.save(
            os.path.join(model_save_dir, "neuron_mrc_list.npy"),
            np.array(neuron_mrc_list, dtype=object)
        )
        torch.save(
            {"masks": new_masks, "statistic": statistic},
            os.path.join(model_save_dir, "neuron_masks.pth")
        )
        logger.info("==> Neuron-level MRC computed and saved, exit.")

        # === 记录神经元选择阶段信息 ===
        experiment_record["selection_phase"] = {
            "num_grad_batches": args.num_grad_batches,
            "neurons_per_layer": args.neurons_per_layer,
            "selected_neurons": selected_neurons_per_layer,
            "mrc_stats": {k: v for k, v in statistic.items()}
        }

        record_path = os.path.join(model_save_dir, "experiment_record.json")
        with open(record_path, "w") as f:
            json.dump(experiment_record, f, indent=4)
        return

    # ======================================================
    # 3) 微调阶段
    # ======================================================
    # 冻结全部参数，只解冻包含 args.layer 名字的部分
    for name, param in model.named_parameters():
        param.requires_grad = True

    # --- 初始性能评估 ---
    _, train_acc_init = evaluate(args, model, trainloader, criterion)
    _, test_acc_init, test_robust_acc_init = validate_robust(
        args, model, testloader, criterion, evalulate_robustness, logger, prefix="Init "
    )
    logger.info(
        "==> Init train acc: {:.2f}%, test acc: {:.2f}%, robust acc: {:.2f}%".format(
            train_acc_init, test_acc_init, test_robust_acc_init
        )
    )
    mask_path = os.path.join(model_save_dir, "neuron_masks.pth")
    mask_ckpt = torch.load(mask_path, map_location=args.device)
    raw_masks = mask_ckpt["masks"]   # 就是你在 cal_neuron_mrc 时保存的 new_masks

    # === 修正参数名前缀 ===
    neuron_masks = {}
    for name, mask in raw_masks.items():
        new_name = re.sub(r"^\d+\.", "", name)
        neuron_masks[new_name] = mask.to(args.device)

    logger.info(f"Loaded {len(neuron_masks)} neuron masks after key alignment.")

    print("Example of neuron_mask keys after cleaning:")
    for k in list(neuron_masks.keys())[:10]:
        print(" ", k)
    # -------------------- 标准微调训练循环 --------------------
    last_robust_acc = None
    record_path = os.path.join(model_save_dir, "experiment_record.json") 
    for epoch in range(start_epoch, start_epoch + args.epochs):

        logger.info("==> Epoch {}".format(epoch))
        logger.info("==> Training...")
        # 1) 训练一轮
        train_loss, train_acc =  training_one_epoch(
            args=args,
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            logger=logger,
            neuron_masks=neuron_masks,
        )
        logger.info("==> Train loss: {:.2f}, train acc: {:.2f}%".format(train_loss, train_acc))

        logger.info("==> Testing (Clean)...")
        # 2) 验证（只在 test 上评估一次即可，需要鲁棒就一起测）
        test_loss, test_acc = validate_clean(
        args, model, testloader, criterion, logger, prefix="[Epoch {}] ".format(epoch)
        )
        logger.info("==> Test loss: {:.2f}, test acc: {:.2f}%".format(test_loss, test_acc))
        if (epoch % args.robust_eval_interval == 0) or (test_acc >= best_acc):
            logger.info(f"==> [Epoch {epoch}] Evaluating Robustness...")
            _, tmp_test_acc, tmp_robust_acc = validate_robust(
                args, model, testloader, criterion, evalulate_robustness, logger,
                prefix=f"[Epoch {epoch}] "
            )
            logger.info(
                f"[Epoch {epoch}] Clean Acc: {tmp_test_acc:.2f}%, Robust Acc: {tmp_robust_acc:.2f}%"
            )
            last_robust_acc = float(tmp_robust_acc)

        # ====== ✅ 就在这里插入“记录本 epoch 结果到文件” ======
        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            # 如果这一轮没有做鲁棒评估，就会是 None
            "robust_acc": None if last_robust_acc is None else float(last_robust_acc),
        }
        experiment_record["results"].append(epoch_record)

        # 每一轮都覆盖写一遍 experiment_record.json，方便中途看曲线
        with open(record_path, "w") as f:
            json.dump(experiment_record, f, indent=4)
        # ====== 插入部分到此结束 ======

        # 3) 保存最好 / 以及周期性 checkpoint
        state = {
            'model': model.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if test_acc > best_acc:
            best_acc = test_acc
            params = "best_params.pth"
            logger.info('==> Saving best params...')
            torch.save(state, model_save_dir + params)
        else:
            if epoch % 2 == 0:
                params = "epoch{}_params.pth".format(epoch)
                logger.info('==> Saving checkpoints...')
                torch.save(state, model_save_dir + params)

        # 4) 更新 scheduler
        scheduler.step()

    # ======================================================
    # 4) 训练结束：加载 best params，做最终评估 + 插值
    # ======================================================
    checkpoint = torch.load(model_save_dir + "best_params.pth")
    model.load_state_dict(checkpoint["model"])

    # 最终评估
    _, final_test_acc, final_robust_acc = validate_robust(
        args, model, testloader, criterion, evalulate_robustness, logger, prefix="Finetune "
    )

    logger.info(
        "==> Finetune final test acc: {:.2f}%, robust acc: {:.2f}%".format(
            final_test_acc, final_robust_acc
        )
    )
    experiment_record["final_result"] = {
        "best_acc": best_acc,
        "final_test_acc": final_test_acc,
        "final_robust_acc": final_robust_acc
    }

    record_path = os.path.join(model_save_dir, "experiment_record.json")
    with open(record_path, "w") as f:
        json.dump(experiment_record, f, indent=4)

    # 插值评估（RiFT 原始流程）
    records = interpolation(
        args, logger, init_sd, deepcopy(model.state_dict()),
        model, testloader, criterion, model_save_dir, evalulate_robustness
    )
    logger.info(records)

    # === ✅ 把插值结果、最终结果都写入 json ===
    experiment_record["interpolation_records"] = records
    experiment_record["final_result"] = {
        "best_acc": float(best_acc),
        "final_test_acc": float(final_test_acc),
        "final_robust_acc": float(final_robust_acc)
    }
    with open(record_path, "w") as f:
        json.dump(experiment_record, f, indent=4)


if __name__ == "__main__":
    main()
