# Author: S.Inoue
# Date: 2/2/2022
# Updated: 2/2/2022
# Project: same compound
# dataset: ChEMBL v.20 kinase

# example code
# python model_mm_gcn_kinase/script/viz_py/A_preprocessing_vizdata.py --model try_1


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
from tqdm import tqdm


'''
path = 'chembl_kinase/data/input/all_seq.txt'
model = 'try_2'

'''

#%%
def main():
    # args
    args = get_args()
    model = args.model
    path = args.seqlist
    # seq_list
    seq_list = set_seq_list(path)

    # make save dir
    data_dir = f'chembl_kinase/viz/{model}/fig/A_process'
    save_dir = f'chembl_kinase/viz/{model}/fig/A_process/sc'
    os.makedirs(save_dir, exist_ok=True)

    # data load
    print('[INFO] load data')
    l_data_seq, l_data_mol, l_data_igs, l_seq_name = [],[],[],[]
    for seq_name in seq_list:
        path_ = f'{data_dir}/mm_gcn_{model}_{seq_name}'
        if os.path.exists(f'{path_}_seq.json') is True: # test dataが存在するときのみ計算
            data_seq_, data_mol_, data_igs_ = data_load(data_dir, model, seq_name)
            l_data_seq.append(data_seq_)
            l_data_mol.append(data_mol_)
            l_data_igs.append(data_igs_)
            l_seq_name.append(seq_name)
        else:
            print(f'# cant load data. maybe no data existed. : {seq_name}')
            pass

    # mol_list
    dict_mol_id = set_mol_id_list(l_data_mol) # 10分かかる
    path = 'chembl_kinase/data/input/all_mol.json'
    save_json(dict_mol_id, path)

    for mol_id in tqdm(dict_mol_id.keys()):
        data_mol_n, data_seq_n, data_igs_n = processing(mol_id, dict_mol_id, l_data_seq, l_data_mol, l_data_igs)
        save_joblib(data_mol_n, f'{save_dir}/mm_gcn_{model}_{mol_id}_seq.jbl')
        save_joblib(data_seq_n, f'{save_dir}/mm_gcn_{model}_{mol_id}_mol.jbl')
        save_joblib(data_igs_n, f'{save_dir}/mm_gcn_{model}_{mol_id}_igs.jbl')


#%%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    parser.add_argument('-m', '--model',
                        type=str,
                        required=True,
                        help='model name -> ex) try_1')
    parser.add_argument('-sn', '--seqname',
                        type=str,
                        required=False,
                        help='sequence name -> ex) LKC_HUMAN')
    parser.add_argument('-sl', '--seqlist',
                        type=str,
                        required=False,
                        help='sequence list file path')
    parser.add_argument('-sd', '--seqdir',
                        type=str,
                        required=False,
                        help='sequence file directory')
    return parser.parse_args()


#%%
def set_seq_list(path):
    # 計算するタンパク質の名前を設定
    seq_list = list(np.loadtxt(path, dtype='str'))
    return seq_list


#%%
def data_load(data_dir, model, seq_name):
    # data_load
    path_ = f'{data_dir}/mm_gcn_{model}_{seq_name}'

    fd = open(f'{path_}_seq.json', mode='r')
    data_seq = json.load(fd)
    fd.close()
    data_mol = joblib.load(f'{path_}_mol.jbl')
    data_igs = joblib.load(f'{path_}_igs.jbl')

    return data_seq, data_mol, data_igs


#%%
def set_mol_id_list(l_data_mol):
    # 全ての固有の化合物IDを取得
    # kinase_chembl = 71580化合物
    l_mol_id_raw_ = [data_mol['mol_id'] for data_mol in l_data_mol]
    l_mol_id_raw = sorted(list(set(sum(l_mol_id_raw_, []))))

    # 化合物ごとに紐づくタンパク質をリスト化
    # dict {k=seq_id, v=mol_ids}
    dict_seq_mol_ids = {data_mol['seq_id']:data_mol['mol_id'] for data_mol in  l_data_mol }
    #
    dict_mol_id = {}
    for mol_id in tqdm(l_mol_id_raw): # 10分かかる
        seq_ids = [seq_id for seq_id, val in dict_seq_mol_ids.items() if mol_id in val]
        dict_mol_id[mol_id] = seq_ids
    return dict_mol_id

#%%
def save_json(data, path):
    tf = open(path, "w")
    json.dump(data, tf, ensure_ascii=False, indent=4, separators=(',', ':'))
    tf.close()
    #print(f'[SAVE] {path}')
    return

def save_joblib(data, path):
    joblib.dump(data, path, compress=3) # compress 圧縮率
    #print(f'[SAVE] {path}')
    return

#%%
def processing(mol_id, dict_mol_id, l_data_seq, l_data_mol, l_data_igs):
    # seq データの順番 index
    d_seq_idx = {data_seq['seq_id']:i for i,data_seq in enumerate(l_data_seq)}
    # データ整理
    data_mol_n, data_seq_n, data_igs_n = {}, {}, {}
    seq_ids = dict_mol_id[mol_id]
    amino_acid_seq, seq_len, target_label,true_label,prediction_score,sum_of_IG,check_score = [],[],[],[],[],[],[]
    mol_IG, seq_IG, seq_IG_z = [],[],[]
    for seq_id in seq_ids:
        seq_idx = d_seq_idx[seq_id]
        mol_idx = l_data_mol[seq_idx]['mol_id'].index(mol_id)
        mol_obj = l_data_mol[seq_idx]['mol_obj'][mol_idx]
        amino_acid_seq.append(l_data_seq[seq_idx]['amino_acid_seq'])
        seq_len.append(l_data_seq[seq_idx]['seq_len'])
        target_label.append(l_data_mol[seq_idx]['target_label'][mol_idx])
        true_label.append(l_data_mol[seq_idx]['true_label'][mol_idx])
        prediction_score.append(l_data_mol[seq_idx]['prediction_score'][mol_idx])
        sum_of_IG.append(l_data_mol[seq_idx]['sum_of_IG'][mol_idx])
        check_score.append(l_data_mol[seq_idx]['check_score'][mol_idx])
        mol_IG.append(l_data_igs[seq_idx]['mol_IG'][mol_idx])
        seq_IG.append(l_data_igs[seq_idx]['seq_IG'][mol_idx])
        seq_IG_z.append(l_data_igs[seq_idx]['seq_IG_z'][mol_idx])

    ## gather data
    data_mol_n['mol_id'] = mol_id
    data_mol_n['seq_id'] = seq_ids
    data_mol_n['mol_obj'] = mol_obj
    #data_mol_n['accuracy'] = accuracy
    #data_mol_n['positive_count'] = positive_count
    #data_mol_n['negative_count'] = negative_count
    #data_mol_n['TP'] = TP
    #data_mol_n['TN'] = TN
    #data_mol_n['FP'] = FP
    #data_mol_n['FN'] = FN

    data_seq_n['mol_id'] = mol_id
    data_seq_n['seq_id'] = seq_ids
    data_seq_n['amino_acid_seq'] = amino_acid_seq
    data_seq_n['seq_len'] = seq_len
    data_seq_n['target_label'] = target_label
    data_seq_n['true_label'] = true_label
    data_seq_n['prediction_score'] = prediction_score
    data_seq_n['sum_of_IG'] = sum_of_IG
    data_seq_n['check_score'] = check_score

    data_igs_n['mol_id'] = mol_id
    data_igs_n['seq_id'] = seq_ids
    data_igs_n['mol_IG'] = mol_IG
    data_igs_n['seq_IG'] = seq_IG
    data_igs_n['seq_IG_z'] = seq_IG_z

    return data_mol_n, data_seq_n, data_igs_n


def data_count(dict_mol_id):
    data_num_ = {key:len(dict_mol_id[key]) for key in dict_mol_id}
    data_num = sorted(data_num_.items(), key = lambda x : x[1], reverse=True)
    head50 = [data_num[i][0] for i in range(50)]


# %%
if __name__ == "__main__":
    main()


    '''
    # .jbl keys
    'amino_acid_seq'    : タンパク質配列 chembl or pdb　要確認!
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
