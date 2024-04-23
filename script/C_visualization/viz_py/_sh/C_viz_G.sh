



SEQ_LIST="KAPCA_HUMAN KP2K2_HUMAN CDK5_HUMAN MK10_HUMAN KPCL_HUMAN"

# visualize G
for SEQ_NAME in $SEQ_LIST; do
    sh script/viz_py/G_corr_acc_plif_residues.sh $SEQ_NAME
done

