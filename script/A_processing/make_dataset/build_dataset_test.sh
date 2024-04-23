#!/bin/sh

# set model
DATASET=$1
MAX_LEN_SEQ=$2

# mkdir
mkdir -p data/dataset/${DATASET}_test
mkdir -p log/${DATASET}/build_dataset


# タンパク質のリスト
DATA_DIR=data/original/${DATASET}
SEQ_LIST=`find ${DATA_DIR} -maxdepth 1 -type d   | sed 's!^.*/!!'`

# build dataset
for SEQ_NAME in $SEQ_LIST; do
    # mkdir
    mkdir -p data/dataset/${DATASET}_test/${SEQ_NAME}

    # run
    kgcn-chem \
    --assay_dir ${DATA_DIR}/${SEQ_NAME} \
    -a 50 \
    --output data/dataset/${DATASET}_test/${SEQ_NAME}/${DATASET}_test_${SEQ_NAME}.jbl \
    --multimodal \
    --no_pseudo_negative \
    --max_len_seq $MAX_LEN_SEQ \
    > log/${DATASET}/build_dataset/build_dataset_test_${SEQ_NAME}.log 2>&1

    #
    mv multimodal_data_index.csv \
    data/dataset/${DATASET}_test/${SEQ_NAME}/multimodal_data_index.csv
done
