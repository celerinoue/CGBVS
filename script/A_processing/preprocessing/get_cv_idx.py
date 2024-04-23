
#%%
import json
import pandas as pd
import sys


#%%
def main():
    # config
    args = sys.argv
    dataset = args[1]
    model = args[2]

    # read_data : info_cv
    with open(f'./result/info/{dataset}/{model}/info_cv.json') as f:
        df1 = json.load(f)
    test_data_idx_ = [df1[i]['test_data_idx'] for i in range(len(df1))]
    cv_idx = {}
    for i in range(len(test_data_idx_)):
        cv_idx.update({k:i+1 for k in test_data_idx_[i]})

    # read data :
    columns = ['mol_idx', 'seq_idx', 'mol_smiles', 'seq_name']
    df2 = pd.read_csv(f'./data/dataset/{dataset}/multimodal_data_index.csv', names=columns)
    df2['seq_name'] = [i.split('/')[-1] for i in df2['seq_name']]

    # concat
    df2['cv_idx'] = df2.index.map(cv_idx)

    # save
    savepath = f"data/dataset/{dataset}/cv_data_index.csv"
    df2.to_csv(savepath)

    return


# %%
if __name__ == '__main__':
    main()
# %%
