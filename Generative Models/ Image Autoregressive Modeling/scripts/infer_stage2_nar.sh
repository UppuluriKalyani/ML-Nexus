#!/bin/bash
ROOT=PATH_TO_YOUR_WORKSPACE
export TORCH_HOME=$ROOT
DATA_ROOT=$ROOT/DiGIT/outputs/dino_base_stage1_cls/results
MODEL_DIR=$ROOT/DiGIT/outputs/dino_base_stage2_nar/checkpoints
RESULTS_PATH=$ROOT/DiGIT/outputs/dino_base_stage2_nar/results
GEN_SUBSET=generate-val-cls
export IMG_SAVE_DIR=${RESULTS_PATH}/gen-${GEN_SUBSET}

mkdir -p $MODEL_DIR
rm -r $IMG_SAVE_DIR
mkdir -p $IMG_SAVE_DIR
cp ./scripts/infer_stage2_nar.sh $MODEL_DIR

devices=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#devices[@]}

for ((i=0; i<$NUM_SHARDS; i++)) do
CUDA_VISIBLE_DEVICES=${devices[$i]} fairseq-generate $DATA_ROOT --user-dir ./fairseq_user \
    --task image_generation_stage2 --source-vocab-size 8192 --target-vocab-size 1024 \
    --path $MODEL_DIR/checkpoint_last.pt --gen-subset $GEN_SUBSET \
    --batch-size 128 --required-batch-size-multiple 1 --fp16 \
    --iter-decode-max-iter 8 --beam 1 --iter-decode-force-max-iter \
    --num-shards $NUM_SHARDS --shard-id $i \
    --results-path ${RESULTS_PATH}/${GEN_SUBSET}_shard_$i &
done
wait

python unified_tsv.py --num-shards $NUM_SHARDS --result-path ${RESULTS_PATH} --subset $GEN_SUBSET

CUDA_VISIBLE_DEVICES=${devices[0]} python fairseq_user/eval_fid.py --results-path $IMG_SAVE_DIR > $RESULTS_PATH/gen-${GEN_SUBSET}_stage1_k200_1.0_step10_fid_is.txt

grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
sed 's/^D-//ig' | sort -nk1 | cut -f3 \
    > ${RESULTS_PATH}/generate-${GEN_SUBSET}.codecs
