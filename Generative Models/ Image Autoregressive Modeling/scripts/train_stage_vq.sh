#!/bin/bash

OUTPUT_DIR=outputs/vq_base
ROOT=PATH_TO_YOUR_WORKSPACE
mkdir -p $OUTPUT_DIR

torchrun --nnodes=$WORLD_SIZE --nproc_per_node=$NPROC_PER_NODE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    $(which fairseq-hydra-train) --config-dir config/stage2 \
    --config-name vq_base \
    hydra.run.dir=$ROOT/DiGIT \
    hydra.output_subdir=$OUTPUT_DIR \
    hydra.job.name=$OUTPUT_DIR/train \
    common.tensorboard_logdir=$OUTPUT_DIR/tb \
    checkpoint.save_dir=$OUTPUT_DIR/checkpoints \
    +task.data=$ROOT/dataset/ILSVRC2012 \
