TRAIN_DATASET=kinase_chembl
TEST_DATASET=kinase_chembl_small_test
MODEL=model_7   #############################
GPU=0
max_epoch=39 # maxのepoch数
POINT=1 # 何epochごとに可視化するか
TIME=$(($max_epoch/$POINT)) # 可視化する回数
VIZ_METHOD=ig # ig, grad_prod, grad, smooth_grad, smooth_ig,
SEQ_LIST="KAPCA_HUMAN LCK_HUMAN KP2K2_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN PLK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN GSK3B_HUMAN KPCA_HUMAN AKT1_HUMAN ABL1_HUMAN PDPK1_HUMAN EGFR_HUMAN IGF1R_HUMAN AURKA_HUMAN CDK1_HUMAN CDK2_HUMAN CHK1_HUMAN INSR_HUMAN KC1D_HUMAN VGFR1_HUMAN SRC_HUMAN VGFR2_HUMAN MK14_HUMAN KSYK_HUMAN"


# train
#sh script/run_py/train_kinase_chembl.sh $TRAIN_DATASET $MODEL $GPU

# test
for SEQ_NAME in $SEQ_LIST; do
    # print
    echo $SEQ_NAME
    # test
    sh script/run_py/test_kinase_chembl.sh $TRAIN_DATASET $TEST_DATASET $MODEL $GPU $POINT $TIME $VIZ_METHOD $SEQ_NAME
done

# visualize A
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

# visualize C
for SEQ_NAME in $SEQ_LIST; do
    python script/viz_py/C_IGheatmap_epoch.py \
    --dataset $TRAIN_DATASET \
    --model $MODEL \
    --epochs $max_epoch \
    --point $POINT \
    --seq_name $SEQ_NAME
done
