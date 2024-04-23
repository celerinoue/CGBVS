# Author: S.Inoue
# Date: 10/25/2021
# Updated: 01/03/2022
# Project: mm_gcn_kinase same protein
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


#%%
def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    epoch = args.epoch
    seq_name = args.seq_name

    th = args.th
    method = args.method
    if args.dtype == 'raw':
        dtype = 'seq_IG'
    elif args.dtype == 'z_score':
        dtype = 'seq_IG_z'
    PDB_ID = args.pdb_id

    # make save dir
    data_dir = f'model_mm_gcn_kinase/viz/fig/{model}/A_process'
    save_dir = f'model_mm_gcn_kinase/viz/fig/{model}/E_featured_residues'
    os.makedirs(save_dir, exist_ok=True)
    # make save dir
    if args.visualize_method == 'ig':
        data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
        hit_res_index_dir = f'viz/{dataset}/{model}/C_IGheatmap/same_protein/{seq_name}'
        save_dir = f'viz/{dataset}/{model}/E_histogram_important_residues/{seq_name}'
    else:
        data_dir = f'viz/{dataset}/{model}_{args.visualize_method}/A_process/{seq_name}'
        hit_res_index_dir = f'viz/{dataset}/{model}_{args.visualize_method}/C_IGheatmap/same_protein/{seq_name}'
        save_dir = f'viz/{dataset}/{model}_{args.visualize_method}/E_histogram_important_residues/{seq_name}'
    os.makedirs(save_dir, exist_ok=True)

    ##=======================================
    ## load data
    data_seq, data_mol, data_igs, hit_res_index = data_load(data_dir, hit_res_index_dir, model, seq_name)
    # IGのdata frameの作成
    df_igs_TP, df_igs_TN = get_df_IG(data_seq, data_mol, data_igs, args.use_raw_ig)
    # 重要残基の抽出
    threshold = args.threshold
    method = args.method
    ## calculating and extracting important residues for prediction
    print(f'[INFO] calculating and extracting important residues for prediction...')
    savepath = f'{save_dir}/E_histogram_important_residues_{seq_name}_{epoch}_{method}_th{threshold}.txt'
    important_residues_index = extract_important_residues(df_igs_TP, df_igs_TN, threshold, method, savepath)

    ##=======================================
    ## fig
    print('[INFO] write fig 1 (IG heatmap)')
    fig1 = IG_HEATMAP()
    savepath = f'{save_dir}/IGheatmap_{seq_name}_{method}_{th}_per_{dtype}.png'
    d0, d1, d2 = fig1.process(df_igs_TP, df_igs_TN, hit_res_index)
    fig1.fig(d0, d1, d2, savepath)

    ## histogram
    print('[INFO] write fig 2 (featured residues histogram)')
    fig2 = HIST_FEATURED_RESIDUES()
    mol, ligand_name = fig2.get_mol(PDB_ID)
    diff_resid_dict = fig2.residue_pick_mean(df_igs_TP, df_igs_TN, th)
    dist_t, dist_f, p_val = fig2.dist_from_ligand(mol, diff_resid_dict, ligand_name)
    savepath = f'{save_dir}/hist_featured_residues_{seq_name}_{method}_{th}_per_{dtype}.png'
    fig2.fig(dist_t, dist_f, p_val, savepath)

    return

# %%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    parser.add_argument('-m', '--model',
                        type=str,
                        required=True,
                        help='seq name (uniprot id)')
    parser.add_argument('-s', '--seqname',
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
    parser.add_argument('-p', '--pdb_id',
                        type=str,
                        required=False,
                        help='pdb_id')

    parser.add_argument('--without_calculate_important_residues', action='store_true',
                        help='without calculating and extracting important residues for prediction')
    parser.add_argument('--use_raw_ig', action='store_true',
                        help='use raw_ig data instead of IG_z_score data')
    return parser.parse_args()


# %%
def data_load(data_dir, hit_res_index_dir, model, seq_name):
    print(f'# load data [{seq_name}]')
    # data_load
    path_ = f'{data_dir}/mm_gcn_{model}_{seq_name}'

    fd = open(f'{path_}_seq.json', mode='r')
    data_seq = json.load(fd)
    fd.close()
    data_mol = joblib.load(f'{path_}_mol.jbl')
    data_igs = joblib.load(f'{path_}_igs.jbl')

    return data_seq, data_mol, data_igs


# %%
def get_df_IG(data_seq, data_mol, data_igs, dtype):
    #====================
    # 対象の化合物を抽出 (条件 : pred_score > 0.7, TP, TN)
    def get_mol_list(data_mol):
        data_ = pd.DataFrame([data_mol['target_label'], data_mol['true_label'], data_mol['prediction_score']],
                            index=['target_label', 'true_label', 'prediction_score'],
                            columns=data_mol['mol_id']).T
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
    def process(self, df_igs_TP, df_igs_TN, hit_res_index):
        # process ==============
        # d0 = TP
        d0 = df_igs_TP
        # d1 = TN
        d1 = df_igs_TN
        # d2 = hit_res_index
        hit_res_dict = {i: 1 for i in hit_res_index} # hitの場所が1のdict
        d_ = {i: 0 for i in range(len(df_igs_TP.T))}  # base 全部0のdict
        d_.update(hit_res_dict)
        d2 = pd.DataFrame.from_dict(d_, orient='index').T
        return d0, d1, d2

    def fig(self, d0, d1, d2, savepath):
        # process ==============

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


class HIST_FEATURED_RESIDUES():
    def get_mol(self, PDB_ID):
        from moleculekit.molecule import Molecule
        print('[INFO] get PDB data...')
        mol = Molecule(PDB_ID)
        mol.filter('chain A')
        # ligand name
        ligand_name = mol.get('resname', sel='not protein')[0]
        return mol, ligand_name

    def residue_pick_mean(self, df_igs_TP, df_igs_TN, th):
        seq_len = len(df_igs_TP.columns)
        # dfの平均のリストを抽出
        arr_TP_IG_mean = np.array(df_igs_TP.mean(axis=0))
        arr_TN_IG_mean = np.array(df_igs_TN.mean(axis=0))
        # 平均の差のランキング上位5%
        diff_IG_sub = np.subtract(arr_TP_IG_mean, arr_TN_IG_mean)
        # 平均値の差とTPの値の積→赤が強いところのみを抽出
        diff_IG_ = np.where(diff_IG_sub < 0, 0, diff_IG_sub) # 0以上のみに抽出
        diff_IG = diff_IG_*arr_TP_IG_mean
        df_diff_IG = pd.DataFrame({'diff_IG': diff_IG})
        # TOPを取得
        #l_diff = np.array(df_diff_IG.sort_values("diff_IG", ascending=False).index)
        #diff_resid_dict = {resid: rank/10 for resid, rank in zip(l_diff, range(len(l_diff), 0, -1))}  # {残基番号:bfactor(順位降順)}
        # head(30)を取得
        head = int(seq_len*th/100)
        l_diff = np.array(df_diff_IG.sort_values("diff_IG", ascending=False).head(head).index)
        fig_th = df_diff_IG.sort_values("diff_IG", ascending=False).iloc[head].values[0]
        diff_resid_dict = {resid: rank for resid, rank in zip(l_diff, range(len(l_diff), 0, -1))}  # {残基番号:bfactor(順位降順)}
        return diff_resid_dict


        # 抽出残基とリガンドの距離 [ リガンド重心座標 - 各残基Ca ]
    def dist_from_ligand(self, mol, diff_resid_dict, ligand_name):
        # ===============================
        # 座標を取得
        def get_coords_prot(mol, diff_resid_dict):  # 蛋白質残基の座標
            # PDBから座標を取得
            resnum_ = np.unique(mol.get('resid', sel='protein'))  # 全残基番号
            coords_ = mol.get('coords', 'name CA')  # 全残基の座標
            coords_cas_dict = {r: c for r, c in zip(resnum_, coords_)}  # dict化
            keys_t = list(coords_cas_dict.keys() &
                          diff_resid_dict.keys())  # 対象の残基番号
            keys_f = list(coords_cas_dict.keys() -
                          diff_resid_dict.keys())  # 対象以外の残基番号
            coords_cas_t = {i: coords_cas_dict[i] for i in keys_t}  # 対象の座標を抽出
            coords_cas_f = {i: coords_cas_dict[i]
                            for i in keys_f}  # 対象以外の座標を抽出
            return coords_cas_t, coords_cas_f

        def get_coords_ligand(mol, ligand_name):  # ligand [重心]
            coords_ligands = mol.get('coords', f'resname {ligand_name}')
            coords_ligand = np.average(coords_ligands, axis=0)  # ligandの重心の座標
            return coords_ligand

        # ===============================
        coords_cas_t, coords_cas_f = get_coords_prot(mol, diff_resid_dict)  # 残基のCA座標
        coords_ligand = get_coords_ligand(mol, ligand_name)  # 対象のligandの重心座標
        # 距離のリスト
        from scipy.spatial import distance
        dist_t = [distance.euclidean(coords_ligand, coords_cas_t[i])
                  for i in list(coords_cas_t.keys())]
        dist_f = [distance.euclidean(coords_ligand, coords_cas_f[i])
                  for i in list(coords_cas_f.keys())]
        # p-val
        from scipy import stats
        p_val = stats.ttest_ind(dist_t, dist_f, equal_var=True)[1]
        return dist_t, dist_f, p_val

    def fig(self, dist_t, dist_f, p_val, savepath):
        ## fig ==================
        # config
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('Pastel1')
        # plot
        fig, ax = plt.subplots(figsize=(12, 8))  # (親グラフ, 子グラフ)
        # kdeplot
        sns.kdeplot(data=dist_t, ax=ax, alpha=0.3, color='purple',fill=True, linewidth=0, label='extracted residues')
        sns.kdeplot(data=dist_f, ax=ax, alpha=0.3, color='gray',fill=True, linewidth=0, label='not-extracted residues')
        # th
        ax.text(0.99, 0.90, f'p-value = {round(p_val, 4)}',va='top', ha='right', transform=ax.transAxes)
        # label
        ax.set_xlabel('euclidean distance between ligand and residues [Å]')
        # title
        #ax.title.set_text(f"Distribution of distance from ligand to TP/TN residues : {target}")
        # show
        plt.legend()
        #plt.show()
        # savefig
        fig.savefig(savepath)
        print(f'[SAVE] {savepath}')
        return

# %%
if __name__ == '__main__':
    main()
# %%
