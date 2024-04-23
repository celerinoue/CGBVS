# Author  : S.Inoue
# Date    : 07/05/2022
# Updated : 07/25/2022

## config
TEST_DATASET=$1
MODEL=$2
SEQ_NAME=$3
MAX_EPOCH=$4 #epochの最大値
PLIF_RESULT=$5
SCORE_CAL_METHOD=$6
RATIO_IG_RES=5
CUTOFF_ROUNDA=1
EPOCH_SEARCH_METHOD=method2

## mkdir
mkdir -p log/$TEST_DATASET/$MODEL/D_search_best_epoch_ratio_res_plif/

## run
python script/viz_py/D_search_best_epoch.py \
--dataset $TEST_DATASET \
--model $MODEL \
--seq_name $SEQ_NAME \
--max_epoch $MAX_EPOCH \
--plif_result $PLIF_RESULT \
--score_cal_method $SCORE_CAL_METHOD \
--ratio_ig_res $RATIO_IG_RES \
--epoch_search_method $EPOCH_SEARCH_METHOD
#--use_raw_ig
#> log/$TEST_DATASET/$MODEL/D_search_best_epoch/D_search_best_epoch_${MODEL}_${SEQ_NAME}.log 2>&1
