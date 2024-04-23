# set model
TRAIN_DATASET=kinase_chembl_small
MODEL=$1
POINT=5 # 何epochごとに可視化するか
max_epoch=200 # maxのepoch数
TIME=40 # 可視化する回数
SEQ_LIST="ABL1_HUMAN AURKA_HUMAN IGF1R_HUMAN PDPK1_HUMAN"


# A
for SEQ_NAME in $SEQ_LIST; do
    for i in `seq $TIME`; do # 実行回数を指定
        # epoch number
        EPOCH_=$((i * $POINT)) # 10epochごとに実行
        EPOCH=$(printf "%05d" $EPOCH_) # zero padding
        echo $EPOCH

        python script/viz_py/A_preprocessing_vizdata.py \
        --dataset $TRAIN_DATASET \
        --model $MODEL \
        --epoch $EPOCH.ckpt \
        --seq_name $SEQ_NAME
    done
done


# C
for SEQ_NAME in $SEQ_LIST; do
    # protein name
        # C
    echo $SEQ_NAME
    # run
    python script/viz_py/C_IGheatmap_epoch.py \
    --dataset $TRAIN_DATASET \
    --model $MODEL \
    --epochs $max_epoch \
    --point $POINT \
    --seq_name $SEQ_NAME
done
