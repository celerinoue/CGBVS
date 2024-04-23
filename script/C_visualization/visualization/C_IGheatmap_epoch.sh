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

## mkdir
mkdir -p log/$TEST_DATASET/$MODEL/C_IGheatmap_epoch/

## run
python script/viz_py/C_IGheatmap_epoch.py \
--dataset $TEST_DATASET \
--model $MODEL \
--epochs $MAX_EPOCH \
--point $POINT \
--seq_name $SEQ_NAME \
> log/$TEST_DATASET/$MODEL/C_IGheatmap_epoch/C_IGheatmap_epoch_${MODEL}_${SEQ_NAME}.log 2>&1

