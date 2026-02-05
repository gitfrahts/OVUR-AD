#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,6
NGPU=2

# evaluate results on RoadAnomaly
torchrun --nproc_per_node=$NGPU --master_port=29500 evaluate.py \
    --score_mode bsl \
    --snapshot pretrained/ra.pth \
    --inference_scale 0.5 0.65 0.85 1.0 1.25 1.75 \
    --inf_temp 1.0 \
    --anomaly_dataset ra \
    --visualize \
    --vis_num 50 \
    --vis_threshold 0.5

# #!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=3,5
# NGPU=2


# # evaluate results on RoadAnomaly
# torchrun --nproc_per_node=$NGPU --master_port=29500 evaluate.py \
#     --score_mode bsl \
#     --snapshot pretrained/ra.pth \
#     --inference_scale 0.5 0.65 0.85 1.0 1.25 1.75 \
#     --inf_temp 1.0 \
#     --anomaly_dataset ra