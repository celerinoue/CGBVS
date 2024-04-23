

TRAIN_DATASET=kinase_chembl
TEST_DATASET=kinase_chembl_small_test
MODEL=model_8   #############################
GPU=0
max_epoch=200 # maxのepoch数
POINT=1 # 何epochごとに可視化するか
TIME=$(($max_epoch/$POINT)) # 可視化する回数
VIZ_METHOD=ig # ig, grad_prod, grad, smooth_grad, smooth_ig,
PDB_ID=6PDJ

#SEQ_LIST="KAPCA_HUMAN LCK_HUMAN KP2K2_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN PLK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN GSK3B_HUMAN KPCA_HUMAN AKT1_HUMAN ABL1_HUMAN PDPK1_HUMAN EGFR_HUMAN IGF1R_HUMAN AURKA_HUMAN CDK1_HUMAN CDK2_HUMAN CHK1_HUMAN INSR_HUMAN KC1D_HUMAN VGFR1_HUMAN SRC_HUMAN VGFR2_HUMAN MK14_HUMAN KSYK_HUMAN"
SEQ_LIST="LCK_HUMAN"
#SEQ_LIST="KAPCA_HUMAN KP2K2_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN"
#SEQ_LIST="PLK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN GSK3B_HUMAN"
#SEQ_LIST="KPCA_HUMAN AKT1_HUMAN ABL1_HUMAN PDPK1_HUMAN EGFR_HUMAN"
#SEQ_LIST="LCK_HUMAN IGF1R_HUMAN AURKA_HUMAN CDK1_HUMAN CDK2_HUMAN"
#SEQ_LIST="INSR_HUMAN KC1D_HUMAN VGFR1_HUMAN SRC_HUMAN"
#SEQ_LIST="CHK1_HUMAN VGFR2_HUMAN MK14_HUMAN KSYK_HUMAN"

# visualize C same protein
for SEQ_NAME in $SEQ_LIST; do
    for i in `seq $TIME`; do # 実行回数を指定
        # epoch number
        EPOCH_=$((i * $POINT)) # 10epochごとに実行
        EPOCH=$(printf "%05d" $EPOCH_) # zero padding
        echo $EPOCH

        python script/viz_py/E_histogram_important_residues_short.py \
        --dataset $TRAIN_DATASET \
        --model $MODEL \
        --epoch $EPOCH.ckpt \
        --seq_name $SEQ_NAME \
        --pdb_id $PDB_ID \
        --visualize_method $VIZ_METHOD \
        --method mdiff \
        --threshold 5
        #--use_raw_ig \

    done
done



# mkdir for log
mkdir -p model_mm_gcn_kinase/log/$model/viz/E_featured_residues



# mkdir for log
mkdir -p model_mm_gcn_kinase/log/$model/viz/E_featured_residues

# run
python model_mm_gcn_kinase/script/viz_py/E_fratured_residues.py \
--model $model \
--seqname $seq_name \
--method $method \
--th $th \
--pdb_id $PDB_ID \
--dtype $dtype \
#> model_mm_gcn_kinase/log/$model/viz/E_featured_residues/E_featured_residues_{$seq}_{$method}_{$th}_{$dtype}.log 2>&1
