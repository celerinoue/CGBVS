# Author  : S.Inoue
# Date    : 07/05/2022
# Updated : 07/05/2022

## config
TEST_DATASET=$1
MODEL=$2
SEQ_NAME=$3
MAX_EPOCH=$4 #epochの最大値
POINT=$5 # 何epochごとに可視化するか
TIME=$(($MAX_EPOCH/$POINT)) # 可視化する回数

## run
for i in `seq $TIME`; do # 実行回数を指定
    # epoch number
    EPOCH_=$((i * $POINT)) # 指定したepochごとに実行
    EPOCH=$(printf "%05d" $EPOCH_) # zero padding
    echo $EPOCH

    #mkdir
    mkdir -p log/$TEST_DATASET/$MODEL/A_preprocessing_vizdata/

    # run
    python script/viz_py/A_preprocessing_vizdata.py \
    --dataset $TEST_DATASET \
    --model $MODEL \
    --epoch $EPOCH.ckpt \
    --seq_name $SEQ_NAME \
    > log/$TEST_DATASET/$MODEL/A_preprocessing_vizdata/vizA_${MODEL}_${EPOCH}_${SEQ_NAME}.log 2>&1

done