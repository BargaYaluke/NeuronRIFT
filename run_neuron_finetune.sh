#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python neuron_rift_mrc_resnet18_cifar10.py \
  --model ResNet18 \
  --dataset TinyImageNet \
  --num_classes 200 \
  --input_size 64 \
  --resume /data/coding/RiFT/ResNet18_Tiny.pth \
  --neurons_per_layer 4 \
  --epochs 10 \
  --batch_size 512 \
  --optim SGDM \
  --lr 0.001 \
  --momentum 0.9 \
  --wd 0.0001 \
  --lr_scheduler cosine \
  --device cuda \
  --mask_mode param \
  --robust_eval_interval 1



