#!/usr/bin/env bash

trap '' HUP

export CUDA_VISIBLE_DEVICES=4,5
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
NGPU=2


torchrun \
    --nproc_per_node=$NGPU \
    --nnodes=1 \
    --node_rank=0 \
    --master_port=12345 \
    train.py \
    --lr 0.00005 \
    --weight_decay 0.0005 \
    --poly_exp 0.9 \
    --class_uniform_pct 0.5 \
    --class_uniform_tile 1024 \
    --crop_size 768 \
    --scale_min 0.5 \
    --scale_max 1.5 \
    --rrotate 0 \
    --max_iter 60000 \
    --bs_mult 2 \
    --gblur \
    --color_aug 0.5 \
    --tag debug \
    --logit_type binary \
    --T 0.07 \
    --adam \
    --tau 0.8 \
    --workdir ./workdir \
    --enable_boundary_suppression False \
    --enable_dilated_smoothing False