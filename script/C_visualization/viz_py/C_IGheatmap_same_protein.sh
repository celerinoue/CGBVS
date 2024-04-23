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
VIZ_METHOD=$6

## mkdir
mkdir -p log/$TEST_DATASET/$MODEL/C_IGheatmap_same_protein/
## rm log if exist
rm -rf log/$TEST_DATASET/$MODEL/C_IGheatmap_same_protein/C_IGheatmap_same_protein_${MODEL}_${SEQ_NAME}.log

## run
fcor i in `seq $TIME`; do # 実行回数を指定
    # epoch number
    EPOCH_=$((i * $POINT)) # 10epochごとに実行
    EPOCH=$(printf "%05d" $EPOCH_) # zero padding
    echo $EPOCH

    python script/viz_py/C_IGheatmap_same_protein.py \
    --dataset $TEST_DATASET \
    --model $MODEL \
    --epoch $EPOCH.ckpt \
    --seq_name $SEQ_NAME \
    --visualize_method $VIZ_METHOD \
    --method mdiff \
    --threshold 5 \
    >> log/$TEST_DATASET/$MODEL/C_IGheatmap_same_protein/C_IGheatmap_same_protein_${MODEL}_${SEQ_NAME}.log 2>&1
    #--without_extract_important_residues \
    #--use_raw_ig \
done
