#!/bin/bash

OUTPUT_DIR=outputs/dino_base_stage2_nar
ROOT=PATH_TO_YOUR_WORKSPACE
mkdir -p $OUTPUT_DIR


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=26667 \
    $(which fairseq-hydra-train) --config-dir config/stage2 \
    --config-name dino_base_nar \
    hydra.run.dir=$ROOT/DiGIT \
    hydra.output_subdir=$OUTPUT_DIR \
    hydra.job.name=$OUTPUT_DIR/train \
    common.tensorboard_logdir=$OUTPUT_DIR/tb \
    checkpoint.save_dir=$OUTPUT_DIR/checkpoints \
    +task.data=$ROOT/dataset/ILSVRC2012 \
        


