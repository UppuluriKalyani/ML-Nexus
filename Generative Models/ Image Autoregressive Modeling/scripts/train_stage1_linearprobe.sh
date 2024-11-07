#!/bin/bash
ROOT=$PATH_TO_YOUR_WORKSPACE
OUTPUT_DIR=outputs/dino_base_stage1_linearprobe_sgd
mkdir -p $OUTPUT_DIR
OUTPUT_DIR=outputs/dino_base_stage1_linearprobe_sgd/lr4e-1_80epoch
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=26663 \
    $(which fairseq-hydra-train) --config-dir config/stage1 \
    --config-name dino_base_linearprobe_sgd \
    hydra.run.dir=$ROOT/DiGIT \
    hydra.output_subdir=$OUTPUT_DIR \
    hydra.job.name=$OUTPUT_DIR/train \
    common.tensorboard_logdir=$OUTPUT_DIR/tb \
    checkpoint.save_dir=$OUTPUT_DIR/checkpoints \
    optimization.lr=[4e-1] \
    +task.data=$ROOT/dataset/ILSVRC2012 \

