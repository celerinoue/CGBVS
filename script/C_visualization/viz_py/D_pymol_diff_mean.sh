



mada






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


# mkdir for log
#mkdir -p model_mm_gcn_kinase/log/$model/viz/D_pymol

# run
python script/viz_py/D_pymol_diff_mean.py \
--dataset $TEST_DATASET \
--model $MODEL \
--seq_name $SEQ_NAME \
--epoch $EPOCH \
--pdb_id $PDB_ID
#> model_mm_gcn_kinase/log/$model/viz/D_pymol/D_pymol_{$seq_name}_{$PDB_ID}.log 2>&1



# viz code
# spectrum b,  yellow_white_purple, minimum=-4.22, maximum=4.22
# spectrum b,  yellow_white_purple
