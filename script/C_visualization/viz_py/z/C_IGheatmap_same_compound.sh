# extract every protein 0~39 28まで流した

#seq_list=`cat model_mm_gcn_kinase/viz/fig/try_2/B_data_statistics/extracted_seq_list_thsatisfied_seq_TP_1000_TN_50_ig_0.1.txt`
#seq_list=LCK_HUMAN
#seq_list=("original" "5aa" "10aa" "E319A" "I314A" "M318A" "T307A" "Y312A")
model=try_2
th=5
# --method mdiff_means, means, mdiff
method=mdiff
# --dtype z_score, raw
dtype=mol_IG

# run
#for seq in $seq_list;
#for mol in '10207821' '11858109' '24955953' '44398371' '5280953' '5283727' '5326843' '9797370' '9869943'
#for mol in '156422' '176155' '216239' '3062316' '123631' '5291' '5494449' '151194' '176870' '208908';
for mol in '10302451' '11048367' '44410720' '447721';
do
    # protein name
    echo $seq
    # run
    python chembl_kinase/script/viz_py/C_IGheatmap_sc.py \
    --model $model \
    --molid $mol \
    --method $method \
    --th $th \
    --dtype $dtype \
    #> chembl_kinase/log/$model/viz/C_IG_heatmap/C_IG_heatmap_{$seq}_{$method}_{$th}_{$dtype}.log 2>&1
done
