#!/bin/bash
ROOT=PATH_TO_YOUR_WORKSPACE
DATA_ROOT=$ROOT/dataset/ILSVRC2012

LAYER=3
CLUSTER=8192

devices=(0 1 2 3)
NUM_SHARDS=${#devices[@]}

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=6666 \
    preprocess/dump_features.py --layer $LAYER \
    --manifest $DATA_ROOT \
    --output dataset/ILSVRC2012/dino_short_224_l$LAYER \
    --layer $LAYER

python preprocess/cluster_kmeans.py \
    --output dataset/ILSVRC2012/dino_short_224_l$LAYER \
    --backend faiss \
    --shard $NUM_SHARDS \
    --cluster-num $CLUSTER

