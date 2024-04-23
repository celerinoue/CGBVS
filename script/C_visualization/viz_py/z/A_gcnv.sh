# set model
TRAIN_DATASET=kinase_chembl_small
MODEL=model_6
POINT=5 # 何epochごとに可視化するか
TIME=2 # 可視化する回数
#SEQ_NAME=LCK_HUMAN # テストを行うタンパク質の名前
# テストしたいタンパク質のリスト
#SEQ_LIST="KAPCA_HUMAN LCK_HUMAN KP2K2_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN PLK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN GSK3B_HUMAN KPCA_HUMAN AKT1_HUMAN ABL1_HUMAN PDPK1_HUMAN EGFR_HUMAN IGF1R_HUMAN AURKA_HUMAN CDK1_HUMAN CDK2_HUMAN CHK1_HUMAN INSR_HUMAN KC1D_HUMAN VGFR1_HUMAN SRC_HUMAN VGFR2_HUMAN MK14_HUMAN KSYK_HUMAN"
SEQ_LIST="KAPCA_HUMAN LCK_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN PLK1_HUMAN KPCT_HUMAN MK09_HUMAN KPCB_HUMAN GSK3B_HUMAN KPCA_HUMAN AKT1_HUMAN"

# run
for SEQ_NAME in $SEQ_LIST; do
    for i in `seq $TIME`; do # 実行回数を指定
        # epoch number
        EPOCH_=$((i * $POINT)) # 10epochごとに実行
        EPOCH=$(printf "%05d" $EPOCH_) # zero padding
        echo $EPOCH

        # run
        for pathfile in viz_raw/$TRAIN_DATASET/$MODEL/$EPOCH.ckpt/$SEQ_NAME/*.jbl; do
            gcnv -i $pathfile
            
        done
    done
done