# Author  : S.Inoue
# Date    : 07/05/2022
# Updated : 07/25/2022

## config
TRAIN_DATASET=kinase_chembl_domain
TEST_DATASET=kinase_chembl_domain_small_test
MODEL=model_8_2
MAX_EPOCH=500 # maxのepoch数
POINT=1 # 何epochごとに可視化するか
VIZ_METHOD=ig # ig, grad_prod, grad, smooth_grad, smooth_ig
PLIF_RESULT="data/other/PLIF/analyzed_kinase_MOE_result2/plif_result.csv"
SCORE_CAL_METHOD=plif_m1 # viz_D plif_m1~3, cmol_m1
## set seq_list
#SEQ_LIST="VGFR2_HUMAN GSK3B_HUMAN CDK1_HUMAN AURKA_HUMAN ERBB2_HUMAN"
#SEQ_LIST="LCK_HUMAN CDK5_HUMAN KPCA_HUMAN KPCE_HUMAN KPCI_HUMAN"
#SEQ_LIST="CDK2_HUMAN KAPCA_HUMAN SRC_HUMAN EGFR_HUMAN"
#SEQ_LIST="CDK19_HUMAN KPCB_HUMAN KPCL_HUMAN KPCT_HUMAN"

# 可視化GOOD AURKA_HUMAN CDK2_HUMAN EGFR_HUMAN 気になる GSK3B_HUMAN KPCA_HUMAN
#SEQ_LIST="AURKA_HUMAN CDK2_HUMAN "
#SEQ_LIST="EGFR_HUMAN GSK3B_HUMAN KPCA_HUMAN"

#PDB_ID="1Y6A"

#SEQ_LIST=$1
#SEQ_LIST="EGFR_HUMAN GSK3B_HUMAN KPCA_HUMAN"


#SEQ_LIST="CDK1_HUMAN LCK_HUMAN"
#SEQ_LIST="CDK2_HUMAN AURKA_HUMAN VGFR2_HUMAN"

#SEQ_LIST="SRC_HUMAN CDK5_HUMAN ERBB2_HUMAN KAPCA_HUMAN"
#SEQ_LIST="KPCI_HUMAN KPCL_HUMAN KPCT_HUMAN"

#SEQ_LIST="CDK2_HUMAN KAPCA_HUMAN SRC_HUMAN EGFR_HUMAN VGFR2_HUMAN GSK3B_HUMAN CDK1_HUMAN AURKA_HUMAN ERBB2_HUMAN LCK_HUMAN CDK5_HUMAN KPCA_HUMAN KPCE_HUMAN KPCI_HUMAN CDK19_HUMAN KPCB_HUMAN KPCL_HUMAN KPCT_HUMAN"
#SEQ_LIST="CDK2_HUMAN KAPCA_HUMAN SRC_HUMAN EGFR_HUMAN VGFR2_HUMAN GSK3B_HUMAN CDK1_HUMAN AURKA_HUMAN KPCL_HUMAN"
SEQ_LIST="ERBB2_HUMAN LCK_HUMAN CDK5_HUMAN KPCA_HUMAN KPCE_HUMAN KPCI_HUMAN CDK19_HUMAN KPCB_HUMAN KPCT_HUMAN"
#SEQ_LIST="CDK2_HUMAN"
##====================================


## run
for SEQ_NAME in $SEQ_LIST; do
echo $SEQ_NAME

## A : processing [必須] [**epochごとに実行**]
#sh script/viz_py/A_preprocessing_vizdata.sh $TEST_DATASET $MODEL $SEQ_NAME $MAX_EPOCH $POINT
echo "[DONE] A_preprocessing_vizdata.sh"

## C : IG_heatmap [**epochごとに実行**]
#sh script/viz_py/C_IGheatmap_same_protein.sh $TEST_DATASET $MODEL $SEQ_NAME $MAX_EPOCH $POINT $VIZ_METHOD
echo "[DONE] C_IGheatmap_same_protein.sh"

## C : IG_heatmap_epoch [**化合物ごとに実行**]
#sh script/viz_py/C_IGheatmap_epoch.sh $TEST_DATASET $MODEL $SEQ_NAME $MAX_EPOCH $POINT
echo "[DONE] C_IGheatmap_epoch.sh"

##========
# 以下は立体構造データが必要

## D : search best epoch [必須]
sh script/viz_py/D_search_best_epoch.sh $TEST_DATASET $MODEL $SEQ_NAME $MAX_EPOCH $PLIF_RESULT $SCORE_CAL_METHOD
echo "[DONE] D_search_best_epoch.sh"


## D : pymol


## 一つずつ可視化する場合

## 可視化したいepochのリストを使用する場合
#sh script/viz_py/D_pymol_diff_mean_useepochlist.sh

## E :




done





'''
MG,
TPO
EDO
SO4
PO4
CSO
ACT
GOL

PDBから原子数とってくる
'''
