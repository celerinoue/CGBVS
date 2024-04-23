#!/bin/sh

# set model
DATASET_NAME=$1
MAX_LEN_SEQ=$2

# mkdir
mkdir -p data/dataset/$DATASET_NAME
mkdir -p log/$DATASET_NAME/build_dataset

## make test dataset
kgcn-chem \
--assay_dir data/original/$DATASET_NAME/ \
-a 50 \
--output data/dataset/${DATASET_NAME}/${DATASET_NAME}.jbl \
--multimodal \
--max_len_seq $MAX_LEN_SEQ \
> log/$DATASET_NAME/build_dataset/build_dataset_train.log 2>&1

# data_idx
mv multimodal_data_index.csv ./data/dataset/${DATASET_NAME}/multimodal_data_index.csv


# --assay_num_limit 100 \ # タンパク質あたりの化合物数の制限値
