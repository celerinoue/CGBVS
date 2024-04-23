# set dir
#cd /data_st02/drug/inoue/CGBVS/

# attach shell
#source ~/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
#conda activate cgbvs


# set model
TRAIN_DATASET=$1
MODEL=$2
GPU=$3

# mkdir
mkdir -p model/$TRAIN_DATASET/$MODEL
mkdir -p result/$TRAIN_DATASET/$MODEL
mkdir -p result/visualization/A_raw_data/$TRAIN_DATASET/$MODEL
mkdir -p log/$TRAIN_DATASET/train/$MODEL

# set dir
sed -e "s/sample_dataset/$TRAIN_DATASET/" -e "s/sample_model/$MODEL/" \
./setting/config/$MODEL.json \
> ./setting/config/tmp/tmp_$MODEL.json


# run
kgcn train \
--config setting/config/tmp/tmp_$MODEL.json \
--dataset data/dataset/$TRAIN_DATASET/$TRAIN_DATASET.jbl \
--gpu $GPU \
> log/$TRAIN_DATASET/train/$MODEL/train.log 2>&1
