#!/bin/bash
# remember to add --save when you want to save the experiment log
# ./run_align.sh 4 zh_en QuatE --save

GPU=$1

DATASET=$2
MODEL=$3
ARGS="$4 $5 $6 $7"
ALPHA=(0.3)
LAYER=(2)
SEED=(12306)

if [ ${DATASET} == "all" ]
then
    DATASETS=("zh_en" "ja_en" "fr_en")
else
    DATASETS=(${DATASET})
fi

if [ ${MODEL} == "TransH" ]
then
    dim="100"
elif [ ${MODEL} == "TransD" ]
then
    dim="100"
else
    dim="200"
fi

for dataset in "${DATASETS[@]}"
do
    for alpha in "${ALPHA[@]}"
    do
        for layer in "${LAYER[@]}"
        do
            for seed in "${SEED[@]}"
            do
                option="--dataset ${dataset} --dim ${dim} --mode ${MODEL} --learning_rate 0.01
                        --alpha ${alpha} --beta ${alpha} --layer ${layer} --rel_update
                        --epochs 20000 --randomseed ${seed} ${ARGS}"
                cmd="CUDA_VISIBLE_DEVICES=${GPU} python train_align.py ${option}"
                echo $cmd
                eval $cmd
            done
        done
    done
done
