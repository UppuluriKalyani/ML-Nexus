#!/bin/bash
ROOT=PATH_TO_YOUR_WORKSPACE
export TORCH_HOME=$ROOT
DATA_ROOT=$ROOT/dataset/ILSVRC2012
MODEL_DIR=$ROOT/DiGIT/outputs/vq_base/checkpoints
RESULTS_PATH=$ROOT/DiGIT/outputs/vq_base/results
GEN_SUBSET=val
export IMG_SAVE_DIR=${RESULTS_PATH}/gen-${GEN_SUBSET}

mkdir -p $MODEL_DIR
rm -r $IMG_SAVE_DIR
mkdir -p $IMG_SAVE_DIR
cp ./scripts/infer_stage_vq.sh $MODEL_DIR

devices=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#devices[@]}

for ((i=0; i<$NUM_SHARDS; i++)) do
CUDA_VISIBLE_DEVICES=${devices[$i]} fairseq-generate $DATA_ROOT --user-dir ./fairseq_user \
    --seed $i --task image_generation_stage2 --source-vocab-size 8192 --target-vocab-size 1024 \
    --augmentation noaug --fp16 \
    --path $MODEL_DIR/checkpoint_last.pt --gen-subset $GEN_SUBSET \
    --batch-size 256 --required-batch-size-multiple 1 \
    --max-len-b 300 --min-len 256 \
    --sampling --sampling-topp 300 --temperature 1.0 --beam 1 \
    --num-shards $NUM_SHARDS --shard-id $i \
    --results-path ${RESULTS_PATH}/${GEN_SUBSET}_shard_$i &
done
wait

python unified_tsv.py --num-shards $NUM_SHARDS --result-path ${RESULTS_PATH} --subset $GEN_SUBSET
echo done

grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
sed 's/^D-//ig' | sort -nk1 | cut -f3 \
    > ${RESULTS_PATH}/generate-${GEN_SUBSET}.codecs

CUDA_VISIBLE_DEVICES=${devices[0]} python fairseq_user/eval_fid.py --results-path $IMG_SAVE_DIR 

