#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python neuron_rift_mrc_resnet18_cifar10.py \
  --model ResNet18 \
  --dataset TinyImageNet \
  --num_classes 200 \
  --input_size 64 \
  --resume /data/coding/RiFT/ResNet18_Tiny.pth \
  --cal_neuron_mrc \
  --neurons_per_layer 4 \
  --num_grad_batches 10 \
  --adv_source train \
  --adv_subset 5000 \
  --adv_seed 0 \
  --epsilon 0.1 \
  --device cuda \
  --contrastive \
  --lambda_con 0.02 \
  --temperature 0.1 \
  --epochs 10 \
  --lr 0.001 \
  --wd 0.0001
