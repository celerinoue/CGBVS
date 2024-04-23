#!/bin/sh

## set model
DATASET_NAME=test-data
MAX_LEN_SEQ=4128
#MODEL=


## build train dataset
#sh ./script/A_processing/make_dataset/build_dataset_train.sh $DATASET_NAME $MAX_LEN_SEQ


## cross validation
#sh ./script/A_processing/make_dataset/split_cv.sh $DATASET_NAME $MAX_LEN_SEQ $MODEL

## visualization
## データセットの中身 可視化のためのコードとか


## ==========================
## build test dataset
sh ./script/A_processing/make_dataset/build_dataset_test.sh $DATASET_NAME $MAX_LEN_SEQ
