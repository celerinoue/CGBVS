# set model
model=try_2
seqlist='chembl_kinase/data/input/all_seq.txt'

# mkdir for log
mkdir -p model_mm_gcn_kinase/log/$model/viz/B_data_statistics

# run
python model_mm_gcn_kinase/script/viz_py/B_data_statistics.py \
--model $model \
--seqlist $seqlist \
> model_mm_gcn_kinase/log/$model/viz/B_data_statistics/B_data_statistics.log 2>&1
