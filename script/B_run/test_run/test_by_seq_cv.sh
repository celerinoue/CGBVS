#!/bin/bash

# config
TRAIN_DATASET=$1
MODEL=$2
GPU=$3
POINT=$4 # 何epochごとに可視化するか
TIME=$5 # 可視化する回数
VIZ_METHOD=ig
SEQ_NAME=$6 # タンパク質名を外部入力

# get_cv_idx
#python script/A_processing/data_analyzer/get_cv_idx.py $TRAIN_DATASET $MODEL
python script/B_run/test_run/make_mol_id_list.py $TRAIN_DATASET $SEQ_NAME

# run
for i in `seq $TIME`; # 実行回数を指定
do
    for CV_ in `seq 5`; # 実行回数を指定
    do
        # cv number
        CV=$(printf "%03d" $CV_) # zero padding
        MOLID_LIST=data/dataset/${TRAIN_DATASET}_test/${SEQ_NAME}/cv/visualize_mol_id_cv${CV_}.txt
        #MOLID_LIST=data/dataset/${TRAIN_DATASET}_test/${SEQ_NAME}/visualize_mol_id.txt
        echo $MOLID_LIST

        # epoch number
        EPOCH_=$((i * $POINT)) # 10epochごとに実行
        EPOCH=$(printf "%05d" $EPOCH_) # zero padding
        MODEL_NUM=$CV.$EPOCH
        echo $MODEL_NUM

        # set config
        sed -e "s/sample_dataset/$TRAIN_DATASET/" \
        -e "s/sample_test_dataset/${TRAIN_DATASET}_test/"\
        -e "s/sample_model/$MODEL/" \
        -e "s/sample_ckpt/$MODEL_NUM.ckpt/" \
        -e "s/sample_seq/$SEQ_NAME/"\
        setting/config/$MODEL.json > setting/config/tmp/tmp_${TRAIN_DATASET}_test_${MODEL}_${SEQ_NAME}_${MODEL_NUM}.json

        #mkdir
        mkdir -p log/${TRAIN_DATASET}_test/$MODEL/test/$SEQ_NAME/

        # run
        kgcn visualize \
        --config setting/config/tmp/tmp_${TRAIN_DATASET}_test_${MODEL}_${SEQ_NAME}_${MODEL_NUM}.json \
        --dataset data/dataset/${TRAIN_DATASET}_test/$SEQ_NAME/${TRAIN_DATASET}_test_${SEQ_NAME}.jbl \
        --gpu $GPU \
        --visualize_method $VIZ_METHOD \
        --visualize_num_list $MOLID_LIST \
        > log/${TRAIN_DATASET}_test/$MODEL/test/$SEQ_NAME/test_${MODEL}_${MODEL_NUM}_${SEQ_NAME}.log 2>&1

        # rm tmp file
        rm -rf setting/config/tmp/tmp_${TRAIN_DATASET}_test_${MODEL}_${SEQ_NAME}_${MODEL_NUM}.json

    done
done




# 詳細はこちら
# https://github.com/clinfo/kGCN/blob/4324cc23d27fc1754fd8e1b3b350d2cf3504015d/kgcn/preprocessing/README.chem.md
