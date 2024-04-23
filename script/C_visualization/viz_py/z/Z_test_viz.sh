# set model
ORIGINAL_DATA=kinase_chembl
TRAIN_DATASET=kinase_chembl_small
TEST_DATASET=test_kinase_chembl_small
MODEL=model_7
N=199 # テストを行うモデルのepochの実行回数
GPU=0
SEQ_NAME=CDK5_HUMAN # テストを行うタンパク質の名前
TH=5
METHOD=mdiff


#============================
# make dataset
# run
#python script/run_py/split_data.py \
#--model $ORIGINAL_DATA \
#--mol_num_limit 50 \
#--for_test_data_true \
#--make_small_data_true \
#--seq_name $SEQ_NAME

# make jbl
#mkdir -p data/dataset/${TEST_DATASET}/${SEQ_NAME}

#kgcn-chem \
#--assay_dir data/original/${TEST_DATASET}/${SEQ_NAME} \
#-a 50 \
#--output data/dataset/${TEST_DATASET}/${SEQ_NAME}/${TEST_DATASET}_${SEQ_NAME}.jbl \
#--multimodal \
#--max_len_seq 4128

#mv multimodal_data_index.csv data/dataset/${TEST_DATASET}/${SEQ_NAME}


#==========================
# run
for i in `seq $N`; # 実行回数を指定
do
    # epoch number
    EPOCH_=$((i * 10)) # 10epochごとに実行
    EPOCH=$(printf "%05d" $EPOCH_) # zero padding
    echo $EPOCH

    #===========

    # set config
    sed -e "s/sample_dataset/$TRAIN_DATASET/" \
    -e "s/sample_model/$MODEL/" \
    -e "s/sample_ckpt/$EPOCH.ckpt/" \
    -e "s/sample_seq/$SEQ_NAME/"\
    config/$MODEL.json > config/tmp/tmp_$MODEL.json

    # test
    #mkdir -p log/$TRAIN_DATASET/$MODEL/test/

    kgcn visualize \
    --config config/tmp/tmp_$MODEL.json \
    --dataset data/dataset/$TEST_DATASET/$SEQ_NAME/${TEST_DATASET}_${SEQ_NAME}.jbl \
    --gpu $GPU \
    > log/$TRAIN_DATASET/$MODEL/test/test_${MODEL}_${EPOCH}_${SEQ_NAME}.log 2>&1

    # A processing
    python script/viz_py/A_preprocessing_vizdata.py \
    --dataset $TRAIN_DATASET \
    --model $MODEL \
    --epoch $EPOCH.ckpt \
    --seq_name $SEQ_NAME \

    # C IG heatmap
    python script/viz_py/C_IGheatmap.py \
    --dataset $TRAIN_DATASET \
    --model $MODEL \
    --epoch $EPOCH.ckpt \
    --seq_name $SEQ_NAME \
    --th $TH \
    --method $METHOD \

done
