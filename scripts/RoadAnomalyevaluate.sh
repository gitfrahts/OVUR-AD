#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,2,3
NGPU=3


# evaluate results on RoadAnomaly
torchrun --nproc_per_node=$NGPU --master_port=12345 evaluate.py \
    --score_mode bsl \
    --snapshot ./pretrained/ra.pth \
    --inference_scale 0.5 0.65 0.85 1.0 1.25 1.75 \
    --inf_temp 1.0 \
    --anomaly_dataset ra

