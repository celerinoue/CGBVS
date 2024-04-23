# Author: S.Inoue
# Date: 10/25/2021
# Updated: 03/01/2022
# Project: same protein
# dataset: ChEMBL v.20 kinase

# example code
# python model_mm_gcn_kinase/script/viz_py/C_IG_heatmap.py --model try_1 -s LCK_HUMAN --method mdiff_means --th 1 --dtype z_score > model_mm_gcn_kinase/log/viz/C_IG_heatmap.log 2>&1
# -> sh model_mm_gcn_kinase/script/viz_py/C_IGheatmap.sh

# %%
import numpy as np
import pandas as pd
import argparse
import json
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec


'''
dataset = "kinase_chembl_small"
model = "kinase_chembl_model_1"
epochs = 200
point = 5
seq_name = "LCK_HUMAN"
'''

# %%


def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    epochs = args.epochs
    point = args.point
    seq_name = args.seq_name

    # =======================================
    # load data
    l_data_seq, l_data_mol, l_data_igs = [], [], []
    l_epoch = [str(i).zfill(5) for i in range(point, epochs-point) if i % point == 0]
    for epoch in l_epoch:
        data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
        save_dir = f'viz/{dataset}/{model}/C_IGheatmap/time_series_by_epoch/{seq_name}'
        os.makedirs(save_dir, exist_ok=True)
        data_seq, data_mol, data_igs = data_load(data_dir, seq_name, epoch)
        l_data_seq.append(data_seq)
        l_data_mol.append(data_mol)
        l_data_igs.append(data_igs)

    # IGのdata frameの作成
    l_mol_id = l_data_mol[0]['mol_id']
    seq_len = l_data_seq[0]['seq_len']
    for mol_num in range(l_data_seq[0]['data_num']):
        df_igs_mol, df_igs_mol_zscore, df_igs_seq, df_igs_seq_zscore = get_df_IG(l_data_mol, l_data_igs, l_epoch, mol_num, seq_len)
        # =======================================
        # fig
        print('[INFO] write fig 1 (IG heatmap)')
        fig1 = IG_HEATMAP()
        savepath = f'{save_dir}/C_IGheatmap_ts-by-epoch_{seq_name}_{l_mol_id[mol_num]}_mol.png'
        fig1.fig(df_igs_mol, savepath)
        savepath = f'{save_dir}/C_IGheatmap_ts-by-epoch_{seq_name}_{l_mol_id[mol_num]}_mol_zscore.png'
        fig1.fig(df_igs_mol_zscore, savepath)
        savepath = f'{save_dir}/C_IGheatmap_ts-by-epoch_{seq_name}_{l_mol_id[mol_num]}_seq.png'
        fig1.fig(df_igs_seq, savepath)
        savepath = f'{save_dir}/C_IGheatmap_ts-by-epoch_{seq_name}_{l_mol_id[mol_num]}_seq_zscore.png'
        fig1.fig(df_igs_seq_zscore, savepath)

        # fig2
        fig2 = SCORE_MAP()
        df_pred_score, df_ans_label = fig2.processing(l_data_mol, l_epoch, mol_num)
        savepath = f'{save_dir}/C_IGheatmap_ts-by-epoch_{seq_name}_{l_mol_id[mol_num]}_answer.png'
        fig2.fig(df_pred_score, df_ans_label, savepath)


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
    parser.add_argument('-e', '--epochs',
                        type=int,
                        required=True,
                        help='')
    parser.add_argument('-p', '--point',
                        type=int,
                        required=True,
                        help='')
    parser.add_argument('-s', '--seq_name',
                        type=str,
                        required=True,
                        help='ex) LCK_HUMAN')
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


# %%
def get_df_IG(l_data_mol, l_data_igs, l_epoch, mol_num, seq_len):
    #化合物原子数
    l_atom_len = [l_data_mol[epoch]['mol_obj'][mol_num].GetNumAtoms() for epoch in range(len(l_epoch))]
    # IGのdfを作成
    igs_mol = [l_data_igs[epoch]['mol_IG'][mol_num] for epoch in range(len(l_epoch))]
    igs_mol_zscore = [l_data_igs[epoch]['mol_IG_z'][mol_num] for epoch in range(len(l_epoch))]
    igs_seq = [l_data_igs[epoch]['seq_IG'][mol_num] for epoch in range(len(l_epoch))]
    igs_seq_zscore = [l_data_igs[epoch]['seq_IG_z'][mol_num] for epoch in range(len(l_epoch))]
    df_igs_mol = pd.DataFrame(igs_mol, index=l_epoch)
    df_igs_mol_zscore = pd.DataFrame(igs_mol_zscore, index=l_epoch)
    df_igs_seq = pd.DataFrame(igs_seq, index=l_epoch).iloc[:,:seq_len].iloc[:, :seq_len]
    df_igs_seq_zscore = pd.DataFrame(igs_seq_zscore, index=l_epoch).iloc[:, :seq_len]

    return df_igs_mol, df_igs_mol_zscore, df_igs_seq, df_igs_seq_zscore

# %%
class IG_HEATMAP():
    def fig(self, df, savepath):
        # process ==============

            # fig ================
        # config
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Pastel1')
        # plot
        fig = plt.figure(figsize=(48, 10))
        gs = gridspec.GridSpec(1, 1, height_ratios=[48])
        ax0 = plt.subplot(gs[0])
        # heatmap
        sns.heatmap(data=df, ax=ax0, cmap='seismic', center=0, cbar=False)
        # lim
        ax0.set_ylim()
        # label
        ax0.axes.xaxis.set_visible(False)
        #ax0.set_ylabel('positive CPI')
        # title
        #fig.suptitle(f"IG heatmap [{seq_name}, {th}% ]", size=15)
        #ax1.title.set_text(f"IGs [True Positive data]")
        #ax2.title.set_text(f"IGs [True Negative data]")
        # plt.legend()

        # save
        # plt.show()
        fig.savefig(savepath)
        print(f'[SAVE] {savepath}')

        return


# %%
class SCORE_MAP():
    def processing(self, l_data_mol, l_epoch, mol_num):
        # pred_score
        l_pred_score = [l_data_mol[epoch]['prediction_score'][mol_num] for epoch in range(len(l_epoch))]
        df_pred_score = pd.DataFrame(l_pred_score, index=l_epoch, columns=[l_data_mol[0]['mol_id'][mol_num]])
        # true_label, ans_label [0=TN, 1=TP, -1=FP, 2=FN]
        l_ans_label = [(l_data_mol[epoch]['true_label'][mol_num] - l_data_mol[epoch]['target_label'][mol_num] + l_data_mol[epoch]['true_label'][mol_num]) for epoch in range(len(l_epoch))]
        df_ans_label = pd.DataFrame(l_ans_label, index=l_epoch, columns=[l_data_mol[0]['mol_id'][mol_num]])
        return df_pred_score, df_ans_label

    def fig(self, df_pred_score, df_ans_label, savepath):
        # config
        sns.set()
        sns.set_style('whitegrid')
        #sns.set_palette('Pastel1')
        # config2
        import matplotlib as mpl
        mpl.rcParams['axes.xmargin'] = 0 # 軸の余白を消す
        # plot
        fig = plt.figure(figsize=(24, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[10,1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        # heatmap
        #sns.heatmap(data=df, ax=ax0, cmap='seismic', center=0)
        sns.lineplot(data=df_pred_score, ax=ax0, color='red', linewidth=2)
        sns.heatmap(data=df_ans_label.T, ax=ax1, cmap='rainbow', center=0.5, vmin=-1, vmax=2, cbar=False, linewidth=.5)
        # lim
        ax0.set_ylim(0.5, 1.0)
        # label
        ax0.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        #ax0.set_ylabel('positive CPI')
        # title
        #fig.suptitle(f"IG heatmap [{seq_name}, {th}% ]", size=15)
        #ax0.title.set_text(f"IGs [True Positive data]")
        #ax2.title.set_text(f"IGs [True Negative data]")
        # plt.legend()

        # save
        # plt.show()
        fig.savefig(savepath)
        print(f'[SAVE] {savepath}')
        return


# %%
if __name__ == '__main__':
    main()
# %%
