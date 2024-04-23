#!/bin/sh

# set model
DATASET_NAME=$1
MAX_LEN_SEQ=$2
MODEL=$3

# mkdir
mkdir -p data/dataset/$DATASET_NAME/cv
mkdir -p log/$DATASET_NAME/build_dataset


# set config
sed -e "s/sample_dataset/$DATASET_NAME/" -e "s/sample_model/$MODEL/" \
./setting/config/$MODEL.json > ./setting/config/tmp/tmp_$MODEL.json


## cross validation dataset
kgcn-cv-splitter \
--config ./setting/config/tmp/tmp_$MODEL.json \
--dataset data/dataset/$DATASET_NAME/$DATASET_NAME.jbl \
--cv_path data/dataset/$DATASET_NAME/cv \
--fold 5 \
> log/$DATASET_NAME/build_dataset/split_cv.log 2>&1
