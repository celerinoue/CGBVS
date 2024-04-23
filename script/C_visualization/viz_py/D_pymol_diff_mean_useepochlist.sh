#config
TRAIN_DATASET=kinase_chembl
MODEL=$3   #############################

###
SEQ_NAME=$1
PDB_ID=$2


# mkdir for log
#mkdir -p model_mm_gcn_kinase/log/$model/viz/D_pymol

# run
python script/viz_py/D_pymol_diff_mean_svepoch.py \
--dataset $TRAIN_DATASET \
--model $MODEL \
--seq_name $SEQ_NAME \
--pdb_id $PDB_ID
#> model_mm_gcn_kinase/log/$model/viz/D_pymol/D_pymol_{$seq_name}_{$PDB_ID}.log 2>&1



# viz code
# spectrum b,  yellow_white_purple, minimum=-4.22, maximum=4.22
# spectrum b,  yellow_white_purple
