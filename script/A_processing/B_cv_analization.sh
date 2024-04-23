#!/bin/bash



## set model
DATASET_NAME=test-data



# get cv_idx
# cvのインデックスをdatasetフォルダに保存 "cv_data_index.csv"
sh ./script/A_processing/data_analyzer/get_cv_idx.py --dataset $DATASET_NAME

# 



# test_dataのみ予測
