# Author: S.Inoue
# Date: 01/04/2022
# Updated: 01/04/2022
# Project: mm_gcn_kinase same compound
# dataset: ChEMBL v.20 kinase

# example code
# python model_mm_gcn_kinase/script/viz_py/B_data_statistics.py --model try_1  > model_mm_gcn_kinase/log/viz/B_data_statistics.log 2>&1
# -> sh model_mm_gcn_kinase/script/viz_py/B_data_statistics.sh


#%%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import argparse
from tqdm import tqdm


'''
path = 'chembl_kinase/data/input/all_mol.json'
model = 'try_2'

'''

#%%
def main():
    # set seq list
    args = get_args()
    model = args.model
    path = args.mollist
    mol_list = load_mol_list(path)

    # make save dir
    data_dir = f'chembl_kinase/viz/{model}/fig/A_process/sc'
    save_dir = f'chembl_kinase/viz/{model}/fig/B_data_statistics/sc'
    os.makedirs(save_dir, exist_ok=True)

    # load data
    print('[INFO] load data')
    l_data_seq, l_data_mol, l_data_igs, l_mol_id = [],[],[],[]
    for mol_id in tqdm(mol_list.keys()):
        path_ = f'{data_dir}/mm_gcn_{model}_{mol_id}'
        if os.path.exists(f'{path_}_seq.jbl') is True: # test dataが存在するときのみ計算
            data_seq_, data_mol_, data_igs_ = data_load(data_dir, model, mol_id)
            l_data_seq.append(data_seq_)
            l_data_mol.append(data_mol_)
            l_data_igs.append(data_igs_)
            l_mol_id.append(mol_id)
        else:
            print(f'# cant load data. maybe no data existed. : {mol_id}')
            pass


    #================================
    # fig 1
    print('[INFO] write fig 1 (data_count_TP_TN)')
    fig1 = COUNT_DATASET()
    savepath = f'{save_dir}/data_count_TP_TN.png'
    th_TP, th_TN = 5, 5   # thを設定 !!!!!!!!!!!
    df, th_mol_list = fig1.process(l_data_mol, th_TP, th_TN)
    print(f'# extracted seq list = {th_mol_list}')
    np.savetxt(f'{save_dir}/extracted_seq_list_datacount_TP_{th_TP}_TN_{th_TN}.txt', th_mol_list, fmt='%s')
    fig1.fig(df, th_TP, th_TN, savepath)


    # fig 2
    print('[INFO] write fig 2 (scat_sum_igs_sep)')
    fig2 = SCAT_IGS()
    gravity = []
    for (seq_name, data_igs) in zip(l_seq_name, l_data_igs):
        savepath = f'{save_dir}/scat_sum_igs_sep_{seq_name}.png'
        df, g_ = fig2.process(data_igs)
        gravity.append(g_)
        try:
            fig2.fig(df, savepath)
        except:
            print(f'# cant write fig : {seq_name}')

    # fig 3
    print('[INFO] write fig 3 (scat_sum_igs_all)')
    fig3 = SCAT_IGS_ALL()
    savepath = f'{save_dir}/scat_sum_igs_all.png'
    th_seq, th_mol = 0.1, 0.3
    df, th_seq_list_seq, th_seq_list_mol = fig3.process(gravity, l_seq_name, th_seq, th_mol)
    print(f'# extracted seq list = {th_seq_list_seq}')
    np.savetxt(f'{save_dir}/extracted_seq_list_IG_seq_TP_ig_{th_seq}.txt', th_seq_list_seq, fmt='%s')
    print(f'# extracted seq list = {th_seq_list_mol}')
    np.savetxt(f'{save_dir}/extracted_seq_list_IG_mol_TP_{th_mol}.txt', th_seq_list_mol, fmt='%s')
    fig3.fig(df, savepath)

    # seq list
    extracted_seq_list_seq = list(set(th_seq_list) & set(th_seq_list_seq))
    np.savetxt(f'{save_dir}/extracted_seq_list_thsatisfied_seq_TP_{th_TP}_TN_{th_TN}_ig_{th_seq}.txt', extracted_seq_list_seq, fmt='%s')
    extracted_seq_list_mol = list(set(th_seq_list) & set(th_seq_list_mol))
    np.savetxt(f'{save_dir}/extracted_seq_list_thsatisfied_mol_TP_{th_TP}_TN_{th_TN}_ig_{th_mol}.txt', extracted_seq_list_mol, fmt='%s')

    return






#%%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    parser.add_argument('-m', '--model',
                        type=str,
                        required=True,
                        help='model name -> ex) try_1')
    parser.add_argument('-l', '--mollist',
                        type=str,
                        required=True,
                        help='path mol_list')
    return parser.parse_args()


#%%
def load_mol_list(path):
    fd = open(path, mode='r')
    mol_list = json.load(fd)
    return mol_list


#%%
def data_load(data_dir, model, mol_id):
    # data_load
    path_ = f'{data_dir}/mm_gcn_{model}_{mol_id}'
    data_mol = joblib.load(f'{path_}_mol.jbl')
    data_seq = joblib.load(f'{path_}_seq.jbl')
    data_igs = joblib.load(f'{path_}_igs.jbl')

    return data_seq, data_mol, data_igs


#%%
# fig 1
class COUNT_DATASET():
    def process(self, l_data_mol, th_TP, th_TN):
        # extract data
        mol_id, TP, TN = [],[],[]
        for data_mol in l_data_mol:
            l_true_label = data_mol['true_label']
            l_target_label = data_mol['target_label']
            mol_id_ = data_mol['mol_id']
            TP_ = sum((true == target)&(target==1) for true,target in zip(l_true_label,l_target_label))
            TN_ = sum((true == target)&(target==0) for true,target in zip(l_true_label,l_target_label))
            mol_id.append(mol_id_)
            TP.append(TP_)
            TN.append(TN_)
        df = pd.DataFrame([TP, TN], index=['TP', 'TN'], columns=mol_id).T
        # th を満たしているseqのリスト
        th_mol_list = list(df[(df['TP'] > th_TP) & (df['TN'] > th_TN)].index)
        return df, th_mol_list

    def fig(self, df, th_TP, th_TN, savepath):
        print('[INFO] write fig...')
        # process
        TP_max = df['TP'].max()
        TN_max = df['TN'].max()

        # fig ==================
        # config
        sns.set()
        sns.set_style('whitegrid')
        #sns.set_palette('bwr')
        # plot
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.scatterplot(ax=ax, data=df, x='TP', y='TN')
        ax.set(xlim=(0, TP_max), ylim=(0, TN_max))
        # set line
        plt.vlines(th_TP, 0, TN_max, "blue")
        plt.hlines(th_TN, 0, TP_max, "blue")
        # label
        ax.set_xlabel('Data Count (true positive)')
        ax.set_ylabel('Data Count (true negative)')
        #fig.suptitle(f"data count [TP/TN]")
        fig.tight_layout()
        # save
        fig.savefig(savepath)
        print(f'[SAVE] {savepath}')
        return


#%%
# fig 2
class SCAT_IGS():
    def process(self, data_igs):
        mol_id = data_igs['mol_id']
        # igs_sum
        mol_igs_sum = [sum(data_igs['mol_IG'][i]) for i in range(len(mol_id))]
        seq_igs_sum = [sum(data_igs['seq_IG'][i]) for i in range(len(mol_id))]
        mol_id = data_igs['mol_id']
        #
        df = pd.DataFrame([mol_igs_sum, seq_igs_sum], index=['mol_igs_sum', 'seq_igs_sum'], columns=mol_id).T
        # 重心点の計算
        g = np.array([np.mean(df['mol_igs_sum']), np.mean(df['seq_igs_sum'])])
        return df, g

    def fig(self, df, savepath):
        # process
        lim_ax = np.abs([df['mol_igs_sum'], df['seq_igs_sum']]).max()

        # fig 1 ==================
        # config
        sns.set()
        sns.set_style('whitegrid')
        #sns.set_palette('bwr')
        # plot
        fig = sns.jointplot(data=df, x='mol_igs_sum', y='seq_igs_sum', xlim=[-lim_ax, lim_ax], ylim=[-lim_ax, lim_ax], height=12) # reg

        # label
        fig.set_axis_labels('sum of igs (mol)', 'sum of igs (protein)', fontsize=12)
        #fig.fig.suptitle(f"Distribution of IGs (mol_IGs vs seq_IGs) [{seq_name}]")
        fig.fig.tight_layout()

        # save
        fig.savefig(savepath)
        print(f'[SAVE] {savepath}')
        return


#%%
# fig 3
class SCAT_IGS_ALL():
    def process(self, gravity, l_seq_name, th_seq, th_mol):
        df = pd.DataFrame(gravity, columns=['mol_igs_ave', 'seq_igs_ave'], index=l_seq_name)
        th_seq_list_seq = list(df[df['seq_igs_ave'] > th_seq].index)
        th_seq_list_mol = list(df[df['mol_igs_ave'] > th_mol].index)
        return df, th_seq_list_seq, th_seq_list_mol

    def fig(self, df, savepath):
        # process
        lim_ax = np.abs([df['mol_igs_ave'], df['seq_igs_ave']]).max()

        # fig 1 ==================
        # config
        sns.set()
        sns.set_style('whitegrid')
        #sns.set_palette('bwr')

        # plot
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.scatterplot(ax=ax, data=df, x='mol_igs_ave', y='seq_igs_ave')
        ax.set(xlim=(-lim_ax, lim_ax), ylim=(-lim_ax, lim_ax))

        # line
        #p_LCK = np.array(df.loc['LCK_HUMAN'])
        #ax.text(p_LCK[0]+0.02, p_LCK[1], 'LCK_HUMAN')
        #ax.scatter(p_LCK[0], p_LCK[1], marker='.', color='red')

        # label
        ax.set_xlabel('average score of igs (mols)')
        ax.set_ylabel('average score of igs (sequences)')
        #fig.suptitle(f"Distribution of IGs (mol_IGs vs seq_IGs) [all protein]")
        fig.tight_layout()

        # save
        fig.savefig(savepath)
        print(f'[SAVE] {savepath}')
        return


# %%
if __name__ == "__main__":
    main()
