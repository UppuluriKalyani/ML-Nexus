#!/bin/bash
ROOT=PATH_TO_YOUR_WORKSPACE
export TORCH_HOME=$ROOT
DATA_ROOT=$ROOT/dataset/ILSVRC2012
MODEL_DIR=$ROOT/DiGIT/outputs/dino_base_stage1_cls/checkpoints
RESULTS_PATH=$ROOT/DiGIT/outputs/dino_base_stage1_cls/results
GEN_SUBSET=val

mkdir -p $MODEL_DIR
cp ./scripts/infer_stage1_ar_classcond.sh $MODEL_DIR

devices=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#devices[@]}

for ((i=0; i<$NUM_SHARDS; i++)) do
CUDA_VISIBLE_DEVICES=${devices[$i]} fairseq-generate $DATA_ROOT --user-dir ./fairseq_user \
    --task image_generation_stage1 --source-vocab-size -1 --target-vocab-size 8192 \
    --seed $i --augmentation noaug --fp16 \
    --path $MODEL_DIR/checkpoint_last.pt --gen-subset $GEN_SUBSET \
    --batch-size 128 --required-batch-size-multiple 1 \
    --max-len-b 300 --min-len 256 \
    --sampling --sampling-topk 300 --temperature 1.0 --beam 1 \
    --num-shards $NUM_SHARDS --shard-id $i \
    --results-path ${RESULTS_PATH}/${GEN_SUBSET}_shard_$i &
done
wait

python unified_tsv.py --num-shards $NUM_SHARDS --result-path ${RESULTS_PATH} --subset $GEN_SUBSET
echo done

grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
sed 's/^D-//ig' | sort -nk1 | cut -f3 \
    > ${RESULTS_PATH}/generate-${GEN_SUBSET}-cls.codecs
