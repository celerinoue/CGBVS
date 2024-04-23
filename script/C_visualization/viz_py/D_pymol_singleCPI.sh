#config
TRAIN_DATASET=kinase_chembl
MODEL=model_7   #############################
VIZ_METHOD=ig # ig, grad_prod, grad, smooth_grad, smooth_ig,

###
SEQ_NAME=$1
PDB_ID=$2
EPOCH= # maxのepoch数




dataset=kinase_chembl_small
model=kinase_chembl_model_6
seq_name=MK09_HUMAN
epoch=00070
PDB_ID=3NPC
ligand_id=10195333
#data_idx=4965

# mkdir for log
#mkdir -p model_mm_gcn_kinase/log/$model/viz/D_pymol

# run
python script/viz_py/D_pymol.py \
--dataset $TRAIN_DATASET \
--model $MODEL \
--seqname $SEQ_NAME \
--epoch $EPOCH \
--pdb_id $PDB_ID \
--ligand_id $ligand_id \
#--data_idx $data_idx \
#> model_mm_gcn_kinase/log/$model/viz/D_pymol/D_pymol_{$seq_name}_{$PDB_ID}.log 2>&1



# gcnv
# 化合物のIGを得たい時
# task番号を入れる
#gcnv -i model_mm_gcn_kinase/viz/viz_data/$model/task_$seq_name/mol_1798_task_0_active_all_scaling.jbl
