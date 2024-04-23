TRAIN_DATASET=kinase_chembl
TEST_DATASET=kinase_PDBbind_refined
MODEL=model_8   #############################
GPU=0
MAX_EPOCH=200 # maxのepoch数
POINT=1 # 何epochごとに可視化するか
TIME=$(($MAX_EPOCH/$POINT)) # 可視化する回数
VIZ_METHOD=ig # ig, grad_prod, grad, smooth_grad, smooth_ig,

# OK
# "LCK_HUMAN CDK5_HUMAN KAPCA_HUMAN KPCL_HUMAN MK10_HUMAN PLK1_HUMAN"
#SEQ_LIST="AKT1_HUMAN EGFR_HUMAN GSK3B_HUMAN"
#SEQ_LIST="INSR_HUMAN KC1D_HUMAN KPCA_HUMAN"
#SEQ_LIST="ABL1_HUMAN PDPK1_HUMAN VGFR1_HUMAN"

#SEQ_LIST="KPCT_HUMAN IGF1R_HUMAN AURKA_HUMAN"
#SEQ_LIST="SRC_HUMAN KSYK_HUMAN"
#SEQ_LIST="KPCB_HUMAN CDK1_HUMAN CDK2_HUMAN"
#SEQ_LIST="VGFR2_HUMAN MK14_HUMAN"
# 残り

#SEQ_LIST="CDK5_HUMAN LCK_HUMAN KAPCA_HUMAN"
#SEQ_LIST="KPCL_HUMAN MK10_HUMAN PLK1_HUMAN"

#SEQ_LIST="KAPCA_HUMAN KPCL_HUMAN CDK5_HUMAN"

#SEQ_LIST="KAPCA_HUMAN KPCL_HUMAN CDK5_HUMAN"
#SEQ_LIST="PLK1_HUMAN"
#SEQ_LIST="MK10_HUMAN"
SEQ_LIST="sample_seq"

#SEQ_LIST="EGFR_HUMAN GSK3B_HUMAN PDPK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN KSYK_HUMAN"
#SEQ_LIST="KPCA_HUMAN AKT1_HUMAN ABL1_HUMAN IGF1R_HUMAN AURKA_HUMAN CDK1_HUMAN CDK2_HUMAN"
#SEQ_LIST="INSR_HUMAN KC1D_HUMAN VGFR1_HUMAN SRC_HUMAN CHK1_HUMAN VGFR2_HUMAN MK14_HUMAN"


# other
#SEQ_LIST="KAPCA_HUMAN LCK_HUMAN KP2K2_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN PLK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN GSK3B_HUMAN KPCA_HUMAN AKT1_HUMAN ABL1_HUMAN PDPK1_HUMAN EGFR_HUMAN IGF1R_HUMAN AURKA_HUMAN CDK1_HUMAN CDK2_HUMAN CHK1_HUMAN INSR_HUMAN KC1D_HUMAN VGFR1_HUMAN SRC_HUMAN VGFR2_HUMAN MK14_HUMAN KSYK_HUMAN"
#SEQ_LIST="LCK_HUMAN"
#SEQ_LIST="KAPCA_HUMAN KP2K2_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN"
#SEQ_LIST="PLK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN GSK3B_HUMAN"
#SEQ_LIST="KPCA_HUMAN AKT1_HUMAN ABL1_HUMAN PDPK1_HUMAN EGFR_HUMAN"
#SEQ_LIST="LCK_HUMAN IGF1R_HUMAN AURKA_HUMAN CDK1_HUMAN CDK2_HUMAN"
#SEQ_LIST="INSR_HUMAN KC1D_HUMAN VGFR1_HUMAN SRC_HUMAN"
#SEQ_LIST="CHK1_HUMAN VGFR2_HUMAN MK14_HUMAN KSYK_HUMAN"

# visualize A
for SEQ_NAME in $SEQ_LIST; do
    for i in `seq $TIME`; do # 実行回数を指定
        # epoch number
        EPOCH_=$((i * $POINT)) # 10epochごとに実行
        EPOCH=$(printf "%05d" $EPOCH_) # zero padding
        echo $EPOCH

        python script/viz_py/A_preprocessing_vizdata.py \
        --dataset $TEST_DATASET \
        --model $MODEL \
        --epoch $EPOCH.ckpt \
        --seq_name $SEQ_NAME
    done


    # viz C
    for i in `seq $TIME`; do # 実行回数を指定
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
        --threshold 5
        #--without_extract_important_residues \
        #--use_raw_ig \
    done


    # visualize C epoch
    python script/viz_py/C_IGheatmap_epoch.py \
    --dataset $TEST_DATASET \
    --model $MODEL \
    --epochs $MAX_EPOCH \
    --point $POINT \
    --seq_name $SEQ_NAME
done
