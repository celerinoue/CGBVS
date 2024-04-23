# Author: S.Inoue
# Date: 10/25/2021
# Updated: 01/03/2022
# Project: mm_gcn_kinase same protein
# dataset: ChEMBL v.20 kinase

# example code
# python model_mm_gcn_kinase/script/viz_py/C_IG_heatmap.py --model try_1 -s LCK_HUMAN --method mdiff_means --th 1 --dtype z_score > model_mm_gcn_kinase/log/viz/C_IG_heatmap.log 2>&1
# -> sh model_mm_gcn_kinase/script/viz_py/C_IGheatmap.sh

# %%
from re import A
import numpy as np
import pandas as pd
import argparse
import json
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec


#%%
def main():
    # args
    args = get_args()
    model = args.model
    mol_id = args.molid
    th = args.th
    method = args.method
    dtype = args.dtype

    '''
    mol_id, model, th, dtype, method = '156422','try_2', 5, 'mol_IG', 'mdiff'
    '''

    # make save dir
    data_dir = f'chembl_kinase/viz/{model}/fig/A_process/sc'
    save_dir = f'chembl_kinase/viz/{model}/fig/C_IGheatmap/sc'
    os.makedirs(save_dir, exist_ok=True)

    ##=======================================
    ## load data
    data_seq, data_mol, data_igs = data_load(data_dir, model, mol_id)
    # IGのdata frameの作成
    df_igs_TP, df_igs_TN = get_df_IG(data_seq, data_mol, data_igs, dtype)
    # 特徴残基の抽出
    #hit_res_index = extract_residues(df_igs_TP, df_igs_TN, th, method)
    #print(f'# featured residue [n = {len(hit_res_index)}]')
    #print(f'# [{hit_res_index}]')

    ##=======================================
    ## fig
    print('[INFO] write fig 1 (IG heatmap)')
    fig1 = IG_HEATMAP()
    savepath = f'{save_dir}/IGheatmap_{mol_id}_{method}_{th}_per_{dtype}.png'
    d0, d1 = fig1.process(df_igs_TP, df_igs_TN)
    fig1.fig(d0, d1, savepath)

    return

# %%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    parser.add_argument('-m', '--model',
                        type=str,
                        required=True,
                        help='seq name (uniprot id)')
    parser.add_argument('--molid',
                        type=str,
                        required=True,
                        help='seq name (uniprot id)')
    parser.add_argument('-mt', '--method',
                        type=str,
                        required=False,
                        help='chooose method of residue extraction [mdiff, means, mdiff_means]')
    parser.add_argument('-t', '--th',
                        type=int,
                        required=False,
                        help='threshold [1,5,10...]')
    parser.add_argument('-d', '--dtype',
                        type=str,
                        required=True,
                        help='choose dtype [z_score, raw]')
    return parser.parse_args()


# %%
def data_load(data_dir, model, mol_id):
    # data_load
    path_ = f'{data_dir}/mm_gcn_{model}_{mol_id}'
    data_mol = joblib.load(f'{path_}_mol.jbl')
    data_seq = joblib.load(f'{path_}_seq.jbl')
    data_igs = joblib.load(f'{path_}_igs.jbl')

    return data_seq, data_mol, data_igs


# %%
def get_df_IG(data_seq, data_mol, data_igs, dtype):
    #====================
    # 対象の化合物を抽出 (条件 : pred_score > 0.7, TP, TN)
    def get_seq_list(data_mol):
        data_ = pd.DataFrame([data_mol['target_label'], data_mol['true_label'], data_mol['prediction_score']],
                            index=['target_label', 'true_label', 'prediction_score'],
                            columns=data_mol['seq_id']).T
        data = data_[data_['prediction_score'] > 0.7]
        TP = data[(data['target_label'] == 1) & (data['true_label'] == 1)]
        TN = data[(data['target_label'] == 0) & (data['true_label'] == 0)]
        print(f'# data count [TP={len(TP)}, TN={len(TN)}]')
        # extract mol index
        TP_seq_index = list(TP.index)
        TN_seq_index = list(TN.index)
        return TP_seq_index, TN_seq_index
    TP_seq_index, TN_seq_index = get_seq_list(data_mol)

    #====================
    # IGのdfを作成
    igs_TP_ = [data_igs[dtype][data_igs['seq_id'].index(p)] for p in TP_seq_index]
    df_igs_TP = pd.DataFrame(igs_TP_, index=TP_seq_index)
    #df_igs_TP = pd.DataFrame(igs_TP_, index=TP_seq_index).iloc[:,:data_seq['seq_len']]
    igs_TN_ = [data_igs[dtype][data_igs['seq_id'].index(p)] for p in TN_seq_index]
    df_igs_TN = pd.DataFrame(igs_TN_, index=TN_seq_index)
    #df_igs_TN = pd.DataFrame(igs_TN_, index=TN_seq_index).iloc[:,:data_seq['seq_len']]
    return df_igs_TP, df_igs_TN


#%%
'''
def extract_residues(df_igs_TP, df_igs_TN, th=5, method='mdiff'):
    # process
    seq_len = len(df_igs_TP.T)
    mean_igs_TP = [np.mean(df_igs_TP[atom]) for atom in range(seq_len)]
    mean_igs_TN = [np.mean(df_igs_TN[atom]) for atom in range(seq_len)]

    #============================
    if method == 'mdiff':
        # (TP/TN間で平均値の差)が大きい上位x%の残基を表す
        diff_igs_list_ = [(mean_igs_TP[r] - mean_igs_TN[r]) for r in range(seq_len)]
        res_rank = pd.DataFrame(diff_igs_list_, columns=['igs']).sort_values(by='igs', ascending=False)

    elif method == 'means':
        # (TPの平均値)が大きい上位x%の残基を表す
        res_rank = pd.DataFrame(mean_igs_TP, columns=['igs']).sort_values(by='igs', ascending=False)

    elif method == 'mdiff_means':
        # [(TP/TN平均値の差)と(TPの平均値)の積]の値が大きい上位x%の残基を表す
        mdiff_igs_list_ = [(mean_igs_TP[r] - mean_igs_TN[r]) if (mean_igs_TP[r] - mean_igs_TN[r]) > 0 else 0 for r in range(seq_len)] # 負の値を0とする
        means_igs_list_ = [r if r > 0 else 0 for r in mean_igs_TP]  # 負の値を0とする
        # mdiff*means
        res_rank_ = [d*m for (d, m) in zip(mdiff_igs_list_, means_igs_list_)]
        res_rank = pd.DataFrame(res_rank_, columns=['igs']).sort_values(by='igs', ascending=False)
    #============================
    hit_res_index = list(res_rank.head(len(res_rank)*int(th)//100).index)
    return hit_res_index
'''

# %%
class IG_HEATMAP():
    def process(self, df_igs_TP, df_igs_TN):
        # process ==============
        # d0 = TP
        d0 = df_igs_TP
        # d1 = TN
        d1 = df_igs_TN
        return d0, d1

    def fig(self, d0, d1, savepath):
        # process ==============

        # fig ================
        # config
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Pastel1')
        # plot
        fig = plt.figure(figsize=(48, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[20,15])
        ax0 = plt.subplot(gs[0])
        #ax1 = plt.subplot(gs[1])
        # heatmap
        sns.heatmap(data=d0, ax=ax0, cmap='seismic', center=0)
        #sns.heatmap(data=d1, ax=ax1, cmap='seismic', center=0)
        #sns.heatmap(data=d2, ax=ax2, cmap='Greens')
        # lim
        ax0.set_ylim()
        # label
        ax0.axes.xaxis.set_visible(False)
        #ax1.axes.xaxis.set_visible(False)
        #ax2.axes.yaxis.set_visible(False)
        ax0.set_ylabel('positive CPI')
        #ax1.set_ylabel('negative CPI')
        #ax2.set_ylabel('hit residue')
        # title
        #fig.suptitle(f"IG heatmap [{seq_name}, {th}% ]", size=15)
        #ax1.title.set_text(f"IGs [True Positive data]")
        #ax2.title.set_text(f"IGs [True Negative data]")
        #plt.legend()

        # save
        # plt.show()
        fig.savefig(savepath)
        print(f'[SAVE] {savepath}')

        return


# %%
if __name__ == '__main__':
    main()
# %%
