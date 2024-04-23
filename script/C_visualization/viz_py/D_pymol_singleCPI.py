# Author: S.Inoue
# Date: 10/25/2021
# Updated: 07/12/2022

'''
example
model = 'try_2'
seq_name = 'LCK_HUMAN'
PDB_ID = '3BYO'
ligand_id = '11785878'

example 2
model = 'try_2'
seq_name = 'LCK_HUMAN'
PDB_ID = '2OFU'
ligand_id = '11843159'

dataset="kinase_chembl_small"
model="kinase_chembl_model_6"
seq_name="KAPCA_HUMAN"
epoch="00050"
PDB_ID="3AMA"
ligand_id="4965"
'''

# %%
import numpy as np
import pandas as pd
import argparse
import json
import joblib
import os
import scipy.stats as stats
import glob


#%%
def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    seq_name = args.seqname
    epoch = args.epoch
    PDB_ID = args.pdb_id
    ligand_id = args.ligand_id
    #data_idx_ = args.data_idx
    data_idx_ = ''
    #dtype = 'seq_IG_z' uze IG z-score

    # make save dir
    data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
    save_dir = f'viz/{dataset}/{model}/D_write_pymol/{seq_name}/igs_raw'
    os.makedirs(save_dir, exist_ok=True)
    #os.makedirs(f'{save_dir}/igs_featured_residues/', exist_ok=True)

    ## load data
    data_seq, data_mol, data_igs = data_load(data_dir, seq_name, epoch)

    ##=======================================

    # PDB dataの取得
    pdb_data = get_pdb_data(PDB_ID)

    # IGの値をそのまま出力
    pymol_igraw = PYMOL_IGRAW()
    # processing
    data_idx, label, bfactor_arr = pymol_igraw.processing(data_seq, data_mol, data_igs, ligand_id, data_idx_, pdb_data)
    # write PDB
    savepath = f'{save_dir}/igs_raw_{seq_name}_{PDB_ID}_{ligand_id}_{epoch}.ckpt_{label}.pdb'
    pymol_igraw.write_PDB(pdb_data, bfactor_arr, savepath)


    '''
    example
    model = 'try_2'
    seq_name = 'LCK_HUMAN'
    PDB_ID = '3BYO'
    ligand_id = '11785878'

    example 2
    model = 'try_2'
    seq_name = 'LCK_HUMAN'
    PDB_ID = '2OFU'
    ligand_id = '11843159'

    dataset="kinase_chembl_small"
    model="kinase_chembl_model_6"
    seq_name="KAPCA_HUMAN"
    epoch="00050"
    PDB_ID="3AMA"
    ligand_id="4965"
    '''

    return

# %%
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
    parser.add_argument('-s', '--seqname',
                        type=str,
                        required=True,
                        help='seq name (uniprot id)')
    parser.add_argument('-e', '--epoch',
                        type=str,
                        required=True,
                        help='')
    parser.add_argument('-p', '--pdb_id',
                        type=str,
                        required=True,
                        help='PDB_ID ex. 1IM0')
    parser.add_argument('-l', '--ligand_id',
                        type=str,
                        required=False,
                        help='ligand_id')
    parser.add_argument('-i', '--data_idx',
                        type=str,
                        required=False,
                        help='data_idx')
    return parser.parse_args()


# %%
def data_load(data_dir, seq_name, epoch):
    # data_load
    fd = open(f'{data_dir}/vizdata_{seq_name}_{epoch}.ckpt_seq.json', mode='r')
    data_seq = json.load(fd)
    fd.close()
    data_igs = joblib.load(f'{data_dir}/vizdata_{seq_name}_{epoch}.ckpt_igs.jbl')
    data_mol = joblib.load(f'{data_dir}/vizdata_{seq_name}_{epoch}.ckpt_mol.jbl')
    return data_seq, data_mol, data_igs


#%%
def get_pdb_data(PDB_ID):
    from moleculekit.molecule import Molecule
    print('[INFO] get PDB data...')
    pdb_data = Molecule(PDB_ID)
    pdb_data.filter('chain A')
    # ligand name
    #ligand_name = mol.get('resname', sel='not protein')[0] # 正確ではない!!!!!!!!!!!!!!!!!!!!!!!
    return pdb_data


#%%
class PYMOL_IGRAW():
    def processing(self, data_seq, data_mol, data_igs, ligand_id, data_idx_, pdb_data):
        # データのindexを取得
        try:
            data_idx = data_mol['mol_id'].index(ligand_id)
        except:
            print('# ligand_id not found in ligand list')
            try:
                data_idx = data_idx_
            except:
                print('[ERROR] data_idx not found in ligand list')
        # data
        label_ = data_mol['true_label'][data_idx]
        if label_ == 0:
            label = 'inactive'
        elif label_ == 1:
            label = 'active'
        print(f"task num : {data_idx}")
        print(f"true_label : {data_mol['true_label'][data_idx]}")
        print(f"target_label : {data_mol['target_label'][data_idx]}")
        print(f"prediction score : {data_mol['prediction_score'][data_idx]}")

        # IGs embedding
        seq_len = data_seq['seq_len']
        full_igs = data_igs['seq_IG_z'][data_idx]
        resid_list = np.unique(pdb_data.resid)[np.unique(pdb_data.resid) < seq_len]
        igs = np.array([full_igs[i] for i in resid_list])
        resid_igs_dict = {i: ig for i, ig in zip(resid_list, igs)}
        bfactor_arr = np.array([resid_igs_dict.get(i) for i in pdb_data.resid])
        return data_idx, label, bfactor_arr

    def write_PDB(self, pdb_data, bfactor_arr, savepath):
        ## embedding
        pdb_data.set("beta", bfactor_arr)
        pdb_data.write(savepath)
        print('[SAVE] pdb data')
        return


# %%
if __name__ == '__main__':
    main()
