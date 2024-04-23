# Author: S.Inoue
# Date: 05/24/2022
# Updated: 05/24/2022
# Project: mm_gcn_kinase same protein
# dataset: ChEMBL v.20 kinase


#%%
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
dataset="kinase_chembl"
model="model_7"   #############################
max_epoch=200 # maxのepoch数
visualize_method="ig" # ig, grad_prod, grad, smooth_grad, smooth_ig,
seq_name = 'LCK_HUMAN'
method="mdiff"
epoch_ = 5
'''


#%%
def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    max_epoch = args.max_epoch
    seq_name = args.seq_name
    method = args.method


    # load plif data
    path_plif = "data/other/PLIF/kinase_MOE_result2.txt"
    path_id_index = 'data/other/PLIF/query_pdb_uniprot_id.csv'
    df_plif = plif_load(path_plif, path_id_index, seq_name)

    #
    l_epoch, l_mean_val, l_acc = [],[],[]
    for epoch_ in range(1, int(max_epoch)):
        try:
            epoch = str(epoch_).zfill(5)
            print(f"# epoch = {epoch}")
            ##=======================================
            # make save dir
            if args.visualize_method == 'ig':
                data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
                save_dir = f'viz/{dataset}/{model}/G_corr_acc_plif_residues/{seq_name}'
            else:
                pass
            os.makedirs(save_dir, exist_ok=True)

            ##=======================================
            ## load data
            data_seq, data_mol, data_igs = data_load(data_dir, seq_name, epoch)
            # IGのdata frameの作成
            df_igs_TP, df_igs_TN = get_df_IG(data_seq, data_mol, data_igs, args.use_raw_ig)

            ##=======================================
            # PLIF残基, PLIF以外の残基のIG
            dict_IG_mean_diff = get_diff_means(df_igs_TP, df_igs_TN)
            mean_val = cal_IG(df_plif, dict_IG_mean_diff)

            ##=======================================
            # タンパク質ごとのACCを取得
            acc = data_seq['accuracy']

            ##=======================================
            # append
            l_epoch.append(int(epoch))
            l_mean_val.append(mean_val)
            l_acc.append(acc)

        except:
            pass

    ##=======================================
    # 相関係数を計算
    #df = pd.DataFrame([l_mean_val,l_acc], columns=l_epoch, index=["diff_means", "ACC"]).T
    df = pd.DataFrame([l_mean_val,l_acc, l_epoch], columns=l_epoch, index=["diff_means", "ACC", "epoch"]).T
    corr = df['diff_means'].corr(df['ACC']) # 相関係数

    # epochごとのp-val,ACC,相関係数の推移
    savepath = f'{save_dir}/scat_acc_plif_{seq_name}.png'
    fig_scat(df, corr, savepath)
    savepath = f'{save_dir}/line_acc_plif_{seq_name}.png'
    fig_line(df, corr, savepath)
    # save txt
    savepath = f'{save_dir}/acc_plif_by_epoch_{seq_name}.csv'
    df.to_csv(savepath)
    return

# %%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    parser.add_argument('--dataset', type=str, required=True,
                        help='')
    parser.add_argument('--model', type=str, required=True,
                        help='')
    parser.add_argument('--max_epoch', type=str, required=True,
                        help='')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='(e.g. --seq_name LCK_HUMAN)')
    # visualize method
    parser.add_argument('--visualize_method', default='ig', choices=['ig', 'grad_prod', 'grad', 'smooth_grad', 'smooth_ig'],
                        help='(ig, grad_prod, grad, smooth_grad, smooth_ig)')
    # important residues
    parser.add_argument('--method', choices=['mdiff', 'means', 'mdiff_means'], default='mdiff',
                        help='chooose method of residue extraction [mdiff, means, mdiff_means]')
    parser.add_argument('--use_raw_ig', action='store_true',
                        help='use raw_ig data instead of IG_z_score data')
    return parser.parse_args()

#%%
def plif_load(path_plif, path_id_index, seq_name):
    #data1
    data1 = pd.read_csv(path_plif, header=None)
    data1[0].str.split('\t', expand=True)
    data1[["a", "b"]] = data1[0].str.split('\t', expand=True)
    data1[["pdb_ID", "chain", "ligand_ID"]] = data1["a"].str.split('_', expand=True)
    data1[["b_1", "PLIF_ID", "score"]] = data1["b"].str.split('_', expand=True)
    data1[["resnum_uniprot", "res_name", "resnum_pdb"]] = data1["b_1"].str.split(':', expand=True)
    data1 = data1.drop(columns = ["a", "b", "b_1", 0])

    # data2
    data2 = pd.read_csv(path_id_index)
    data2 = data2.rename(columns={"pdbid":"pdb_ID", "db_code":"uniprot_ID"})
    data2 = data2[data2["uniprot_ID"].str.contains("HUMAN")]
    data2 = data2[["pdb_ID", "uniprot_ID"]]

    # merge
    df_plif_all = pd.merge(data1, data2, on='pdb_ID')
    # query
    df_plif = df_plif_all[df_plif_all['uniprot_ID'] == seq_name]
    return df_plif

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
def get_diff_means(df_igs_TP, df_igs_TN):
    seq_len = len(df_igs_TP.columns)
    # dfの平均のリストを抽出
    arr_TP_IG_mean = np.array(df_igs_TP.mean(axis=0))
    arr_TN_IG_mean = np.array(df_igs_TN.mean(axis=0))
    # 平均の差のランキング上位5%
    arr_IG_mean_diff = np.subtract(arr_TP_IG_mean, arr_TN_IG_mean)
    dict_IG_mean_diff = {r:v for r,v in zip(range(1,seq_len+1, 1), arr_IG_mean_diff)}
    return dict_IG_mean_diff

#%%
def cal_IG(df_plif, dict_IG_mean_diff):
    # hit resのindex を取得
    def get_PLIF_IG(df_plif, dict_IG_mean_diff):
        # 対象のPLIFのIG
        keys_t = [int(s) for s in sorted(list(set(df_plif['resnum_uniprot'])))] # pLIF対象の残基のindex
        dict_IG_mean_diff_t = {i: dict_IG_mean_diff[i] for i in keys_t}  # 対象の座標を抽出
        # 対象以外のnonPLIFのIG
        keys_f = list(set(dict_IG_mean_diff.keys()) - set(keys_t))  # 対象以外の残基番号
        dict_IG_mean_diff_f = {i: dict_IG_mean_diff[i] for i in keys_f}  # 対象の座標を抽出
        return dict_IG_mean_diff_t, dict_IG_mean_diff_f

    def cal_diff_mean(dict_IG_mean_diff_t, dict_IG_mean_diff_f):
        mean_dist_t = np.mean(list(dict_IG_mean_diff_t.values()))
        mean_dist_f = np.mean(list(dict_IG_mean_diff_f.values()))
        mean_val = mean_dist_t - mean_dist_f # 平均値の差 (対象残基の化合物距離の平均値 - 対象残基以外の化合物距離の平均値)
        return mean_val

    dict_IG_mean_diff_t, dict_IG_mean_diff_f = get_PLIF_IG(df_plif, dict_IG_mean_diff)
    mean_val = cal_diff_mean(dict_IG_mean_diff_t, dict_IG_mean_diff_f)
    return mean_val

#%%
def fig_scat(df, corr, savepath):
    ## fig ==================
    # config
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Pastel1')
    # plot
    fig, ax = plt.subplots(figsize=(12, 12))  # (親グラフ, 子グラフ)
    # kdeplot
    sns.scatterplot(data=df, x='diff_means', y='ACC', color="green")
    # lim
    ax.set_xlim(-max(np.abs(df['diff_means'])), max(np.abs(df['diff_means'])))
    ax.set_ylim(0.5, 1.0)
    # th
    ax.text(0.99, 0.95, f'corr = {round(corr, 3)}',va='top', ha='right', transform=ax.transAxes)
    # label
    #ax.set_xlabel('euclidean distance between ligand and residues [Å]')
    #ax.invert_xaxis()
    # title
    #ax.title.set_text(f"Distribution of distance from ligand to TP/TN residues : {target}")
    # show
    plt.legend()
    #plt.show()
    # savefig
    fig.savefig(savepath)
    print(f'[SAVE] {savepath}')
    return

#%%
def fig_line(df, corr, savepath):
    ## fig ==================
    # config
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Pastel1')
    # plot
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    # kdeplot
    sns.barplot(data=df, ax=ax1, x="epoch", y="ACC", color="blue", alpha=0.3)
    sns.lineplot(data=df, ax=ax2, x="epoch", y="diff_means", color="red")
    # lim
    ax1.set_ylim((0.6,1))
    ax2.set_ylim(-max(np.abs(df['diff_means'])), max(np.abs(df['diff_means'])))
    ax2.set_xlim(0, 200)
    # th
    ax1.text(0.90, 0.95, f'corr = {round(corr, 3)}',va='top', ha='right', transform=ax2.transAxes)
    # label
    #ax.set_xlabel('euclidean distance between ligand and residues [Å]')
    ax1.axes.xaxis.set_visible(False)
    #ax2.invert_yaxis()

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
