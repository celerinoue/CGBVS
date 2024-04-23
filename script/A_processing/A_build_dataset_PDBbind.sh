#!/bin/sh

# set model
DATA_DIR=data/original/pdbbind-refined
DATASET_NAME=pdbbind-redined
MAX_LEN_SEQ=4128

# フォルダ読み込み, タンパク質リスト
#SEQ_LIST=`find ${DATA_DIR} -maxdepth 1 -type d   | sed 's!^.*/!!'`


#for SEQ_NAME in $SEQ_LIST
SEQ_NAME=1a1e

## make test dataset
#do
echo $SEQ_NAME

sh ./script/A_processing/make_dataset/build_dataset_test_single_prot.sh $DATA_DIR $DATASET_NAME $MAX_LEN_SEQ $SEQ_NAME

#done
