TRAIN_DATASET=kinase_chembl
TEST_DATASET=kinase_chembl_small_test
MODEL=model_7   #############################
GPU=0
MAX_EPOCH=200 # maxのepoch数
VIZ_METHOD=ig # ig, grad_prod, grad, smooth_grad, smooth_ig,

#SEQ_LIST="KAPCA_HUMAN KP2K2_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN"
#SEQ_LIST="PLK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN GSK3B_HUMAN"
#SEQ_LIST="KPCA_HUMAN AKT1_HUMAN ABL1_HUMAN PDPK1_HUMAN EGFR_HUMAN"
#SEQ_LIST="LCK_HUMAN IGF1R_HUMAN AURKA_HUMAN CDK1_HUMAN CDK2_HUMAN"
#SEQ_LIST="INSR_HUMAN KC1D_HUMAN VGFR1_HUMAN SRC_HUMAN"
SEQ_LIST="CHK1_HUMAN VGFR2_HUMAN MK14_HUMAN KSYK_HUMAN"

# visualize E
for SEQ_NAME in $SEQ_LIST; do
    python script/viz_py/G_corr_acc_plif_residues.py \
    --dataset $TRAIN_DATASET \
    --model $MODEL \
    --max_epoch $MAX_EPOCH \
    --seq_name $SEQ_NAME \
    --visualize_method $VIZ_METHOD \
    --method mdiff
    #--use_raw_ig \
done
