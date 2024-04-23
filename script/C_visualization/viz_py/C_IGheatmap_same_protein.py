# Author: S.Inoue
# Date: 10/25/2021
# Updated: 05/09/2022
# Project: same protein
# dataset: ChEMBL v.20 kinase

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

#%%
def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    epoch = args.epoch
    seq_name = args.seq_name

    # make save dir
    if args.visualize_method == 'ig':
        data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
        save_dir = f'viz/{dataset}/{model}/C_IGheatmap/same_protein/{seq_name}'
    else:
        data_dir = f'viz/{dataset}/{model}_{args.visualize_method}/A_process/{seq_name}'
        save_dir = f'viz/{dataset}/{model}_{args.visualize_method}/C_IGheatmap/same_protein/{seq_name}'
    os.makedirs(save_dir, exist_ok=True)

    ##=======================================

    ## processing
    data_seq, data_mol, data_igs = data_load(data_dir, seq_name, epoch)
    ## make IG dataframe
    df_igs_TP, df_igs_TN = get_df_IG(data_seq, data_mol, data_igs, args.use_raw_ig)

    if not args.without_extract_important_residues:
        # args
        threshold = args.threshold
        method = args.method
        ## calculating and extracting important residues for prediction
        print(f'[INFO] calculating and extracting important residues for prediction...')
        savepath = f'{save_dir}/C_IGheatmap_sp_{seq_name}_{epoch}_{method}_th{threshold}.txt'
        try:
            important_residues_index = extract_important_residues(df_igs_TP, df_igs_TN, threshold, method, savepath)

            ## fig
            print('[INFO] write fig (IG heatmap)')
            fig = IG_HEATMAP()
            if not args.use_raw_ig:
                savepath = f'{save_dir}/C_IGheatmap_sp_{seq_name}_{epoch}_{method}_th{threshold}.png'
            else:
                savepath = f'{save_dir}/C_IGheatmap_sp_{seq_name}_{epoch}_{method}_th{threshold}_rawIG.png'
            fig.fig_with_important_residues(df_igs_TP, df_igs_TN, important_residues_index, savepath)
        except:
            print("[ERROR] maybe, number of TN or TP data none...")
            pass

    else:
        print(f'[INFO] without calculating and extracting important residues for prediction')
        print(f'[INFO] save only IG heatmap')
        ## fig
        print('[INFO] write fig (IG heatmap)')
        fig = IG_HEATMAP()
        if not args.use_raw_ig:
            savepath = f'{save_dir}/C_IGheatmap_sp_{seq_name}_{epoch}.png'
        else:
            savepath = f'{save_dir}/C_IGheatmap_sp_{seq_name}_{epoch}_rawIG.png'
        fig.fig_only_ig(df_igs_TP, df_igs_TN, savepath)

    return

# %%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    parser.add_argument('--dataset', type=str, required=True,
                        help='')
    parser.add_argument('--model', type=str, required=True,
                        help='')
    parser.add_argument('--epoch', type=str, required=True,
                        help='')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='(e.g. --seq_name LCK_HUMAN)')
    # visualize method
    parser.add_argument('--visualize_method', default='ig', choices=['ig', 'grad_prod', 'grad', 'smooth_grad', 'smooth_ig'],
                        help='(ig, grad_prod, grad, smooth_grad, smooth_ig)')
    # important residues
    parser.add_argument('--without_extract_important_residues', action='store_true',
                        help='without calculating and extracting important residues for prediction')
    parser.add_argument('--method', choices=['mdiff', 'means', 'mdiff_means'], default='mdiff',
                        help='chooose method of residue extraction [mdiff, means, mdiff_means]')
    parser.add_argument('--threshold', type=str, default='5',
                        help='(e.g. --th 0,5,10)')
    parser.add_argument('--use_raw_ig', action='store_true',
                        help='use raw_ig data instead of IG_z_score data')

    return parser.parse_args()


# %%
def data_load(data_dir, seq_name, epoch):
    # data_load
    fd = open(f'{data_dir}/vizdata_{seq_name}_{epoch}_seq.json', mode='r')
    data_seq = json.load(fd)
    fd.close()
    data_igs = joblib.load(f'{data_dir}/vizdata_{seq_name}_{epoch}_igs.jbl')
    data_mol = joblib.load(f'{data_dir}/vizdata_{seq_name}_{epoch}_mol.jbl')
    return data_seq, data_mol, data_igs


# %%
def get_df_IG(data_seq, data_mol, data_igs, use_raw_ig):
    #====================
    # 対象の化合物を抽出 (条件 : pred_score > 0.7, TP, TN)
    def get_mol_list(data_mol):
        data_ = pd.DataFrame([data_mol['target_label'], data_mol['true_label'], data_mol['prediction_score']],
                            index=['target_label', 'true_label', 'prediction_score'], columns=data_mol['mol_id']).T
        data = data_[data_['prediction_score'] > 0.0]
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
def extract_important_residues(df_igs_TP, df_igs_TN, threshold, method, savepath):
    # process
    seq_len = len(df_igs_TP.T)
    mean_igs_TP = [np.mean(df_igs_TP[res]) for res in range(seq_len)]
    mean_igs_TN = [np.mean(df_igs_TN[res]) for res in range(seq_len)]

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
    important_residues_index = list(res_rank.head(len(res_rank)*int(threshold)//100).index)
    print(f'# important residues (n = {len(important_residues_index)})')

    ## save
    f = open(savepath, 'w')
    for x in important_residues_index:
        f.write(str(x) + "\n")
    f.close()
    print('[SAVE] important residues')

    return important_residues_index


# %%
class IG_HEATMAP():
    def fig_only_ig(self, df_igs_TP, df_igs_TN, savepath):
        # process ==============
        # d0 = TP
        d0 = df_igs_TP
        # d1 = TN
        d1 = df_igs_TN
        # fig ================
        # config
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Pastel1')
        # plot
        fig = plt.figure(figsize=(48, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[20, 15])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        # heatmap
        sns.heatmap(data=d0, ax=ax0, cmap='seismic', center=0)
        sns.heatmap(data=d1, ax=ax1, cmap='seismic', center=0)
        # lim
        ax0.set_ylim()
        # label
        ax0.axes.xaxis.set_visible(False)
        ax1.axes.xaxis.set_visible(False)
        ax0.set_ylabel('positive CPI')
        ax1.set_ylabel('negative CPI')
        # title
        #fig.suptitle(f"IG heatmap [{seq_name}, {th}% ]", size=15)
        #ax0.title.set_text(f"IGs [True Positive data]")
        #ax1.title.set_text(f"IGs [True Negative data]")
        #plt.legend()

        # save
        # plt.show()
        fig.savefig(savepath)
        print(f'[SAVE] {savepath}')
        return

    def fig_with_important_residues(self, df_igs_TP, df_igs_TN, important_residues_index, savepath):
        # process ==============
        # d0 = TP
        d0 = df_igs_TP
        # d1 = TN
        d1 = df_igs_TN
        # d2 = hit_res_index
        # hitの場所が1のdict
        hit_res_dict = {i: 1 for i in important_residues_index}
        d_ = {i: 0 for i in range(len(df_igs_TP.T))}  # base 全部0のdict
        d_.update(hit_res_dict)
        d2 = pd.DataFrame.from_dict(d_, orient='index').T
        # fig ================
        # config
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Pastel1')
        # plot
        fig = plt.figure(figsize=(48, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[20,15,2])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        # heatmap
        sns.heatmap(data=d0, ax=ax0, cmap='seismic', center=0)
        sns.heatmap(data=d1, ax=ax1, cmap='seismic', center=0)
        sns.heatmap(data=d2, ax=ax2, cmap='Greens')
        # lim
        ax0.set_ylim()
        # label
        ax0.axes.xaxis.set_visible(False)
        ax1.axes.xaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ax0.set_ylabel('positive CPI')
        ax1.set_ylabel('negative CPI')
        ax2.set_ylabel('hit residue')
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
