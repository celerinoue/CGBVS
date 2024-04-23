TRAIN_DATASET=kinase_chembl
TEST_DATASET=kinase_PDBbind_general
MODEL=model_8   #############################
GPU=3
MAX_EPOCH=200 # maxのepoch数
POINT=5 # 何epochごとに可視化するか
TIME=$(($MAX_EPOCH/$POINT)) # 可視化する回数
VIZ_METHOD=ig # ig, grad_prod, grad, smooth_grad, smooth_ig,

SEQ_LIST=sample_seq



# visualize C same protein
for SEQ_NAME in $SEQ_LIST; do

    for i in `seq $TIME`; do # 実行回数を指定
        # epoch number
        EPOCH_=$((i * $POINT)) # 10epochごとに実行
        EPOCH=$(printf "%05d" $EPOCH_) # zero padding
        echo $EPOCH

        python script/viz_py/A_preprocessing_vizdata.py \
        --dataset $TEST_DATASET \
        --model $MODEL \
        --epoch $EPOCH.ckpt \
        --seq_name $SEQ_NAME
    done


    for i in `seq $TIME`; do # 実行回数を指定
        # epoch number
        EPOCH_=$((i * $POINT)) # 10epochごとに実行
        EPOCH=$(printf "%05d" $EPOCH_) # zero padding
        echo $EPOCH

        python script/viz_py/C_IGheatmap_same_protein.py \
        --dataset $TEST_DATASET \
        --model $MODEL \
        --epoch $EPOCH.ckpt \
        --seq_name $SEQ_NAME \
        --visualize_method $VIZ_METHOD \
        --method mdiff \
        --threshold 5
        #--without_extract_important_residues \
        #--use_raw_ig \

    done


    # visualize C epoch
    python script/viz_py/C_IGheatmap_epoch.py \
    --dataset $TEST_DATASET \
    --model $MODEL \
    --epochs $MAX_EPOCH \
    --point $POINT \
    --seq_name $SEQ_NAME
done
