TRAIN_DATASET=kinase_chembl_domain
TEST_DATASET=kinase_chembl_domain_small_test
MODEL=model_8   #############################
MAX_EPOCH=200 # maxのepoch数
VIZ_METHOD=ig # ig, grad_prod, grad, smooth_grad, smooth_ig,

SEQ_LIST=$1
PDB_ID=$2

# visualize E
for SEQ_NAME in $SEQ_LIST; do
    python script/viz_py/F_corr_acc_important_residues.py \
    --dataset $TEST_DATASET \
    --model $MODEL \
    --max_epoch $MAX_EPOCH \
    --seq_name $SEQ_NAME \
    --pdb_id $PDB_ID \
    --visualize_method $VIZ_METHOD \
    --method mdiff \
    --threshold $3
    #--use_raw_ig \
done
