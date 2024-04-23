# set dir
cd /data_st02/drug/inoue/CGBVS

# attach shell
#source ~/.pyenv/versions/miniconda-latest/envs/cgbvs/envs/cgbvs
#conda activate cgbvs


##=======================================================
## config
TRAIN_DATASET=$1
MODEL=$2   #############################
GPU=4
MAX_LEN_SEQ=4128
POINT=1 # 何epochごとに可視化するか
TIME=50 # 可視化する回数

# 自動 all
#SEQ_LIST=`find data/original/kinase_chembl -type d -maxdepth 1  | sed 's!^.*/!!'`

# seq_list指定

# test-data
#SEQ_LIST="MYLK_HUMAN CDK6_HUMAN KCC1D_HUMAN"

# chembl_kinase
#SEQ_LIST="LCK_HUMAN"
#SEQ_LIST="CDK19_HUMAN"
SEQ_LIST=$3
#SEQ_LIST="KPCD1_HUMAN"
#SEQ_LIST="M3K14_HUMAN"
#SEQ_LIST="KPCA_HUMAN"
#SEQ_LIST="KPCB_HUMAN"
#SEQ_LIST="KPCL_HUMAN"
#SEQ_LIST="KPCE_HUMAN"
#SEQ_LIST="CDK6_HUMAN"
#SEQ_LIST="KPCI_HUMAN"
#SEQ_LIST="ABL2_HUMAN"
#SEQ_LIST="KPCZ_HUMAN"
#SEQ_LIST="KPCT_HUMAN"
#SEQ_LIST="PDK1_HUMAN"
#SEQ_LIST="KCC1D_HUMAN"

#SEQ_LIST="KAPCG_HUMAN KC1E_HUMAN KAPCB_HUMAN"
#SEQ_LIST="KPCD1_HUMAN M3K14_HUMAN KPCA_HUMAN"
#SEQ_LIST="KPCB_HUMAN KPCL_HUMAN KPCE_HUMAN CDK6_HUMAN"
#SEQ_LIST="KPCI_HUMAN ABL2_HUMAN KPCZ_HUMAN"
#SEQ_LIST="KPCT_HUMAN PDK1_HUMAN KCC1D_HUMAN"


# CVあり
for SEQ_NAME in $SEQ_LIST
do
echo $SEQ_NAME
sh script/B_run/test_run/test_by_seq_cv.sh $TRAIN_DATASET $MODEL $GPU $POINT $TIME $VIZ_METHOD $SEQ_NAME
done
