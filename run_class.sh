#!/bin/bash
# remember to add --save when you want to save the experiment log
# ./run_class.sh 1 fb15k TransE --save

GPU=$1

DATASET=$2
MODEL=$3
ARGS="$4 $5 $6 $7"
DIM=(32)
LR=(0.01)
SEED=(12306)

if [ ${DATASET} == "am" ]
then
    ALPHA=(0.3)
    LAYER=(2)
elif [ ${DATASET} == "wordnet" ]
then
    ALPHA=(0.5)
    LAYER=(2)
elif [ ${DATASET} == "fb15k" ]
then
    ALPHA=(0.5)
    LAYER=(0)
fi


for alpha in "${ALPHA[@]}"
do
    for layer in "${LAYER[@]}"
    do
        for dim in "${DIM[@]}"
        do
            for lr in "${LR[@]}"
            do
                for seed in "${SEED[@]}"
                do
                    option="--dataset ${DATASET} --dim ${dim} --mode ${MODEL} --learning_rate ${lr}
                            --alpha ${alpha} --beta ${alpha} --layer ${layer} --rel_update
                            --epochs 1000 --randomseed ${seed} ${ARGS}"
                    cmd="CUDA_VISIBLE_DEVICES=${GPU} python train_class.py ${option}"
                    echo $cmd
                    eval $cmd
                done
            done
        done
    done
done
