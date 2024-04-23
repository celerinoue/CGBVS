# kinaseドメインのみでCPI予測

TRAIN_DATASET=kinase_chembl
TEST_DATASET=kinase_PDBbind_refined #refined
MODEL=model_8   #############################
GPU=1
MAX_LEN_SEQ=4128

##=======================================================
## test
POINT=1 # 何epochごとに可視化するか
TIME=500 # 可視化する回数
VIZ_METHOD=ig

##=======================================================
# run
for i in `seq $TIME`; # 実行回数を指定
do
    # epoch number
    EPOCH_=$((i * $POINT)) # 5epochごとに実行
    EPOCH=$(printf "%05d" $EPOCH_) # zero padding
    echo $EPOCH

    # set config
    sed -e "s/sample_dataset/$TRAIN_DATASET/" \
    -e "s/sample_test_dataset/$TEST_DATASET/"\
    -e "s/sample_model/$MODEL/" \
    -e "s/sample_ckpt/$EPOCH.ckpt/" \
    config/$MODEL.json > config/tmp/tmp_${TEST_DATASET}_${MODEL}_${EPOCH}.json

    #mkdir
    mkdir -p log/$TEST_DATASET/$MODEL/test/

    # run
    kgcn visualize \
    --config config/tmp/tmp_${TEST_DATASET}_${MODEL}_${EPOCH}.json \
    --dataset data/dataset/${TEST_DATASET}/${TEST_DATASET}.jbl \
    --gpu $GPU \
    --visualize_method $VIZ_METHOD \
    > log/$TEST_DATASET/$MODEL/test/test_${MODEL}_${EPOCH}.log 2>&1

    # rm tmp file
    rm -rf config/tmp/tmp_${TEST_DATASET}_${MODEL}_${EPOCH}.json
done
