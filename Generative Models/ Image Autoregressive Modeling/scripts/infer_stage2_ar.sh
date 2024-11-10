#!/bin/bash
ROOT=PATH_TO_YOUR_WORKSPACE
export TORCH_HOME=$ROOT
DATA_ROOT=$ROOT/DiGIT/outputs/base_8k_stage1/results
MODEL_DIR=$ROOT/DiGIT/outputs/base_8k_stage2_400epoch/checkpoints
RESULTS_PATH=$ROOT/DiGIT/outputs/base_8k_stage2_400epoch/results
GEN_SUBSET=generate-val
export IMG_SAVE_DIR=${RESULTS_PATH}/${GEN_SUBSET}

mkdir -p $MODEL_DIR
rm -r $IMG_SAVE_DIR
mkdir -p $IMG_SAVE_DIR
cp ./scripts/infer_stage2_ar.sh $MODEL_DIR

devices=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#devices[@]}

for ((i=0; i<$NUM_SHARDS; i++)) do
CUDA_VISIBLE_DEVICES=${devices[$i]} fairseq-generate $DATA_ROOT --user-dir ./fairseq_user \
    --task image_generation_stage2 --source-vocab-size 8192 --target-vocab-size 1024 \
    --path $MODEL_DIR/checkpoint_last.pt --gen-subset $GEN_SUBSET \
    --batch-size 128 --required-batch-size-multiple 1 --fp16 \
    --max-len-b 600 --min-len 256 \
    --sampling --sampling-topp 0.8 --temperature 1.0 --beam 1 \
    --num-shards $NUM_SHARDS --shard-id $i \
    --results-path ${RESULTS_PATH}/${GEN_SUBSET}_shard_$i &
done
wait

python unified_tsv.py --num-shards $NUM_SHARDS --result-path ${RESULTS_PATH} --subset $GEN_SUBSET

grep "^D\-" ${RESULTS_PATH}/${GEN_SUBSET}.txt | \
sed 's/^D-//ig' | sort -nk1 | cut -f3 \
    > ${RESULTS_PATH}/${GEN_SUBSET}.codecs

CUDA_VISIBLE_DEVICES=${devices[0]} python fairseq_user/eval_fid.py --results-path $IMG_SAVE_DIR 
