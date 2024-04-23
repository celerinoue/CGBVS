

## config
data_dir=/data_st02/drug/inoue/CGBVS/data/other/PDBbind/refined-set
save_dir=/data_st02/drug/inoue/CGBVS/data/original/pdbbind-refined

## assay_result
assay_index_name_path=/data_st02/drug/inoue/CGBVS/data/other/PDBbind/index/INDEX_refined_name.2020
assay_index_data_path=/data_st02/drug/inoue/CGBVS/data/other/PDBbind/index/INDEX_refined_data.2020
#assay_index_name_path=/data_st02/drug/inoue/CGBVS/data/other/PDBbind/index/INDEX_general_PL_name.2020
#assay_index_data_path=/data_st02/drug/inoue/CGBVS/data/other/PDBbind/index/INDEX_general_PL_data.2020


##==============================
    ## general set
    # List of protein-ligand complexes with known binding data in PDBbind v.2020
    # 19443 complexes in total, clustered by 95% protein sequence similarity

    ## refined set
    # List of protein-ligand complexes in the PDBbind refined set v.2020
    # 5316 complexes in total, clustered by 90% protein sequence similarity
##==============================


## make
python ./script/A_processing/make_dataset/processing_assay.py \
$data_dir $save_dir $assay_index_name_path $assay_index_data_path
