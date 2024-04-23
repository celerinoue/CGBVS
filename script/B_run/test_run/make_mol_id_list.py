
#%%
import pandas as pd
import argparse
import os
import sys


#%%
def main():
    # config
    args = sys.argv
    dataset = args[1]
    seq_name = args[2]


    # get train_data cv_id_list
    df1 = pd.read_csv(f"data/dataset/{dataset}/cv_data_index.csv", index_col=0)
    df1 = df1[df1["seq_name"]==seq_name]
    df1 = df1[['mol_smiles', 'cv_idx']]

    #
    columns = ['mol_idx', 'seq_idx', 'mol_smiles', 'seq_name']
    df2 = pd.read_csv(f"data/dataset/{dataset}_test/{seq_name}/multimodal_data_index.csv", header=None, names=columns)
    df2 = df2[['mol_smiles']]

    # merge
    df = pd.merge(df2, df1, on='mol_smiles', how='left')

    # save
    os.makedirs(f"data/dataset/{dataset}_test/{seq_name}/cv", exist_ok=True)
    for cv in range(1,6):
        savepath = f"data/dataset/{dataset}_test/{seq_name}/cv/visualize_mol_id_cv{str(cv)}.txt"
        l_ = list(df[df['cv_idx'] == cv].index)
        l = [str(i) for i in l_]

        with open(savepath, 'w', encoding='utf-8') as f:
            #f.write(str(x) + ",")
            f.write(','.join(l))

    return




#%%
if __name__ == '__main__':
    main()



'''
dataset = 'test-data'
seq_name = 'CDK6_HUMAN'


'''
