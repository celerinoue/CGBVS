# Author: S.Inoue
# Date: 10/25/2021
# Updated: 01/03/2022
# Project: mm_gcn_kinase same protein
# dataset: ChEMBL v.20 kinase

# %%
import numpy as np
import pandas as pd
import argparse
import json
import joblib
import os

'''
dataset="kinase_chembl"
model="model_7"   #############################
epoch="00150" # maxのepoch数
visualize_method="ig" # ig, grad_prod, grad, smooth_grad, smooth_ig,
PDB_ID = "6PDJ"
seq_name = 'LCK_HUMAN'
method="mdiff"
threshold = 5
'''



#%%
def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    seq_name = args.seq_name
    PDB_ID = args.pdb_id # '6PDJ'

    # make save dir
    data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}/'
    save_dir = f'viz/{dataset}/{model}/D_pymol/diff_mean/{seq_name}/'
    os.makedirs(save_dir, exist_ok=True)

    ## load mol data
    mol, ligand_name = get_mol(PDB_ID)

    ## epoch list
    path = f"viz/{dataset}/{model}/F_corr_acc_important_residues/{seq_name}/top5_epoch_{seq_name}_per5.txt"
    epoch_list = list(np.loadtxt(path))

    for epoch_ in epoch_list:
        ## load data
        epoch = str(int(epoch_)).zfill(5)
        print(f"[INFO] epoch_num = {epoch}")
        data_seq, data_mol, data_igs = data_load(data_dir, seq_name, epoch)

        ##=======================================
        # IG heatmap
        df_igs_TP, df_igs_TN = get_df_IG(data_seq, data_mol, data_igs, False)
        arr_diff_mean_IG = cal_diff_mean_IG(df_igs_TP, df_igs_TN)

        # IGの値をそのまま出力
        write_pymol = WRITE_PYMOL()
        bfactor_arr = write_pymol.processing(data_seq, arr_diff_mean_IG, mol)
        savepath = f'{save_dir}/igs_diff_mean_{seq_name}_{epoch}_{PDB_ID}_{ligand_name}.pdb'
        write_pymol.write_PDB(mol, bfactor_arr, savepath)
    return

# %%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    parser.add_argument('--dataset', type=str, required=True,
                        help='')
    parser.add_argument('--model', type=str, required=True,
                        help='')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='(e.g. --seq_name LCK_HUMAN)')
    parser.add_argument('--pdb_id', type=str, required=True,
                        help='pdb_id')
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
def get_mol(PDB_ID):
    from moleculekit.molecule import Molecule
    print('[INFO] get PDB data...')
    mol = Molecule(PDB_ID)
    mol.filter('chain A')
    mol.remove('resname HOH')
    # ligand name
    ligand_name = mol.get('resname', sel='not protein')[0]

    return mol, ligand_name


# %%
def get_df_IG(data_seq, data_mol, data_igs, use_raw_ig):
    #====================
    # 対象の化合物を抽出 (条件 : pred_score > 0.7, TP, TN)
    def get_mol_list(data_mol):
        data_ = pd.DataFrame([data_mol['target_label'], data_mol['true_label'], data_mol['prediction_score']],
                            index=['target_label', 'true_label', 'prediction_score'], columns=data_mol['mol_id']).T
        data = data_[data_['prediction_score'] > 0.7]
        TP = data[(data['target_label'] == 1) & (data['true_label'] == 1)]
        TN = data[(data['target_label'] == 0) & (data['true_label'] == 0)]
        print(f'# data count [TP={len(TP)}, TN={len(TN)}]')
        # extract mol index
        TP_mol_index = list(TP.index)
        TN_mol_index = list(TN.index)
        return TP_mol_index, TN_mol_index
    TP_mol_index, TN_mol_index = get_mol_list(data_mol)
    #====================
    # IGのdfを作成
    if use_raw_ig == True:
        dtype = 'seq_IG'
    elif use_raw_ig == False:
        dtype = 'seq_IG_z'
    igs_TP_ = [data_igs[dtype][data_igs['mol_id'].index(p)] for p in TP_mol_index]
    df_igs_TP = pd.DataFrame(igs_TP_, index=TP_mol_index).iloc[:,:data_seq['seq_len']]
    igs_TN_ = [data_igs[dtype][data_igs['mol_id'].index(p)] for p in TN_mol_index]
    df_igs_TN = pd.DataFrame(igs_TN_, index=TN_mol_index).iloc[:,:data_seq['seq_len']]
    return df_igs_TP, df_igs_TN

#%%
def cal_diff_mean_IG(df_igs_TP, df_igs_TN):
    #seq_len = len(df_igs_TP.columns)
    # dfの平均のリストを抽出
    arr_TP_IG_mean = np.array(df_igs_TP.mean(axis=0))
    arr_TN_IG_mean = np.array(df_igs_TN.mean(axis=0))
    # 平均の差 TP - TN
    arr_diff_mean_IG = np.subtract(arr_TP_IG_mean, arr_TN_IG_mean)
    return arr_diff_mean_IG

#%%
class WRITE_PYMOL():
    def processing(self, data_seq, arr_diff_mean_IG, mol):
        # use TOP5%
        seq_len = data_seq['seq_len']
        imp_res_num = seq_len*5//100
        important_residues_index = sorted([sorted([(v,i) for (i,v) in enumerate(arr_diff_mean_IG)])[i][1] for i in range(imp_res_num)])
        # IGs embedding
        resid_list = np.unique(mol.resid)[np.unique(mol.resid) < seq_len]
        target_igs_dict = {i:arr_diff_mean_IG[i] for i in important_residues_index}
        igs_dict = {i: 0 for i in resid_list}
        for k,v in target_igs_dict.items():
            if k in igs_dict.keys():
                igs_dict[k+1] = v
            else:
                pass
        bfactor_arr = np.array([igs_dict.get(i) for i in mol.resid])
        return bfactor_arr

    def write_PDB(self, mol, bfactor_arr, savepath):
        ## embedding
        mol.set("beta", bfactor_arr)
        mol.write(savepath)
        print('[SAVE] pdb data')
        return


# %%
if __name__ == '__main__':
    main()
