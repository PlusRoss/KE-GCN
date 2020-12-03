#!/bin/bash
# remember to add --save when you want to save the experiment log
# ./run_rel_align.sh 4 all TransE --save

GPU=$1

DATASET=$2
MODEL=$3
ARGS="$4 $5 $6 $7"
ALPHA=(0.3)
LAYER=(2)
SEED=(12306)
SREL=(0)
WREL=(0.0)

if [ ${DATASET} == "all" ]
then
    DATASETS=("zh_en" "ja_en" "fr_en")
else
    DATASETS=(${DATASET})
fi

for dataset in "${DATASETS[@]}"
do
    for alpha in "${ALPHA[@]}"
    do
        for layer in "${LAYER[@]}"
        do
            for seed in "${SEED[@]}"
            do
                for srel in "${SREL[@]}"
                do
                    for wrel in "${WREL[@]}"
                    do
                        option="--dataset ${dataset} --dim 200 --mode ${MODEL} --learning_rate 0.01
                                --alpha ${alpha} --beta ${alpha} --layer ${layer} --auto --rel_update --epochs 20000
                                --randomseed ${seed} --rel_weight ${wrel} --rel_seed ${srel} ${ARGS}"
                        cmd="CUDA_VISIBLE_DEVICES=${GPU} python train_rel_align.py ${option}"
                        echo $cmd
                        eval $cmd
                   done
                done
            done
        done
    done
done
