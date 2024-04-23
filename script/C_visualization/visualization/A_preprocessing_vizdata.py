# Author: S.Inoue
# Date: 12/8/2021
# Updated: 3/1/2022
# dataset: ChEMBL v.20 kinase

# example code
# python ./script/viz_py/A_preprocessing_vizdata.py --model try_1


#%%
import json
import numpy as np
import pandas as pd
import joblib
import os
import scipy.stats as stats
from glob import glob
import argparse
from multiprocessing.dummy import Pool

'''
dataset = "kinase_chembl_small"
model = "kinase_chembl_epoch2000"
epoch = "00010.ckpt"
seq_name = "LCK_HUMAN"
'''

#%%
def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    epoch = args.epoch
    seq_name = args.seq_name

    # make save dir
    data_dir = f'viz_raw/{dataset}/{model}/{epoch}/{seq_name}'
    save_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
    os.makedirs(save_dir, exist_ok=True)

    # processing
    print('[INFO] viz data loading...')
    #print(f'# seq = {seq_name}')
    with Pool() as p:  # multishread
        files = glob(f'{data_dir}/*.jbl')
        data_list = p.map(joblib.load, files)
        if len(data_list) > 0:
            # data process
            data_seq, data_mol, data_igs = processing(seq_name, data_list)
            # save data
            save_json(data_seq, f'{save_dir}/vizdata_{seq_name}_{epoch}_seq.json')
            save_joblib(data_mol, f'{save_dir}/vizdata_{seq_name}_{epoch}_mol.jbl')
            save_joblib(data_igs, f'{save_dir}/vizdata_{seq_name}_{epoch}_igs.jbl')
        else:
            print('no visualize data')
    return


#%%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    parser.add_argument('-d', '--dataset',
                        type=str,
                        required=True,
                        help='')
    parser.add_argument('-m', '--model',
                        type=str,
                        required=True,
                        help='')
    parser.add_argument('-e', '--epoch',
                        type=str,
                        required=True,
                        help='')
    parser.add_argument('-s', '--seq_name',
                        type=str,
                        required=True,
                        help='ex) LCK_HUMAN')
    return parser.parse_args()


#%%
def processing(seq_name, data_list):
    ## protein
    seq_id = seq_name
    amino_acid_seq = data_list[0]['amino_acid_seq']
    seq_len = len(amino_acid_seq)

    ## mol
    mol_id = [data_list[i]['mol_id'] for i in range(len(data_list))]
    mol_obj = [data_list[i]['mol'] for i in range(len(data_list))]

    ## label & score
    target_label = [data_list[i]['target_label'] for i in range(len(data_list))]
    true_label = [data_list[i]['true_label'] for i in range(len(data_list))]
    prediction_score = [data_list[i]['prediction_score'] for i in range(len(data_list))]
    sum_of_IG = [data_list[i]['sum_of_IG'] for i in range(len(data_list))]
    check_score = [data_list[i]['check_score'] for i in range(len(data_list))]

    ## IGs
    mol_IG = [np.squeeze(data_list[i]['features_IG']).sum(axis=1) for i in range(len(data_list))] # [50,83]
    mol_IG_z = [stats.zscore(i) for i in mol_IG]
    seq_IG = [np.squeeze(data_list[i]['embedded_layer_IG']).sum(axis=1) for i in range(len(data_list))]
    seq_IG_z = [stats.zscore(i) for i in seq_IG]

    ## description
    df = pd.DataFrame([target_label, true_label], index=['target_label', 'true_label']).T
    data_num = len(df)
    accuracy = len(df[df['true_label'] == df['target_label']]) / data_num
    T = df[df['true_label'] == 1]
    positive_count = len(T)
    TP = len(T[T['target_label'] == 1])
    FN = len(T[T['target_label'] == 0])
    N = df[df['true_label'] == 0]
    negative_count = len(N)
    TN = len(N[N['target_label'] == 0])
    FP = len(N[N['target_label'] == 1])

    ## gather data
    data_seq, data_mol, data_igs = {}, {}, {}
    data_seq['seq_id'] = seq_id
    data_seq['amino_acid_seq'] = amino_acid_seq
    data_seq['seq_len'] = seq_len
    data_seq['data_num'] = data_num
    data_seq['accuracy'] = accuracy
    data_seq['positive_count'] = positive_count
    data_seq['negative_count'] = negative_count
    data_seq['TP'] = TP
    data_seq['TN'] = TN
    data_seq['FP'] = FP
    data_seq['FN'] = FN

    data_mol['seq_id'] = seq_id
    data_mol['mol_id'] = mol_id
    data_mol['mol_obj'] = mol_obj
    data_mol['target_label'] = target_label
    data_mol['true_label'] = true_label
    data_mol['prediction_score'] = prediction_score
    data_mol['sum_of_IG'] = sum_of_IG
    data_mol['check_score'] = check_score

    data_igs['mol_id'] = mol_id
    data_igs['mol_IG'] = mol_IG
    data_igs['seq_IG'] = seq_IG
    data_igs['seq_IG_z'] = seq_IG_z
    data_igs['mol_IG_z'] = mol_IG_z
    return data_seq, data_mol, data_igs


def data_check(path):
    import os
    is_file = os.path.isfile(path)
    if is_file:
        print(f"file exist")
    return is_file

def save_json(data, path):
    tf = open(path, "w")
    json.dump(data, tf, ensure_ascii=False, indent=4, separators=(',', ':'))
    tf.close()
    print(f'[SAVE] {path}')
    return


def save_joblib(data, path):
    joblib.dump(data, path, compress=3) # compress 圧縮率
    print(f'[SAVE] {path}')
    return


# %%
if __name__ == "__main__":
    main()


    '''
    # .jbl keys
    'amino_acid_seq'    : タンパク質配列 chembl or pdb 要確認!
    'features'          : 化合物特徴量 (batch_size, feature)  shape (50, 81)
    'features_IG'       : 化合物IG shape (50, 81)
    'adjs'              : 化合物特徴量 隣接行列    (batch_size, feature)  shape (50, 50)
    'adjs_IG'           : 化合物 隣接行列のIG shape (50, 81)
    'embedded_layer'    : embedded_layerの特徴量? (dim, 配列長, アミノ酸) shape(1, 4128, 25)
    'embedded_layer_IG' : embedded_layerのIG
    'mol'               : 化合物 RDKit Mol object
    'mol_smiles'        : 化合物 SMILES 学習に用いたもので、IUPAC登録の正しいものではない
    'mol_id'            : 化合物 pubchem CID
    'prediction_score'  : float 0.5~1
    'target_label'      : int 0 or 1 (1=positive, 0=negative)
    'true_label'        : int 0 or 1
    'check_score'       : float IG(1)-IG(0), an approximate value of a total value of IG
        # If IG is calculated correctly, "total of IG" approximately equal to "difference
        # between the prediction score with scaling factor = 1 and with scaling factor = 0".
    'sum_of_IG'         : IGのsum
    '''

# %%
