# Author: S.Inoue
# Date: 05/24/2022
# Updated: 07/05/2022


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


# %%
def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    max_epoch = args.max_epoch
    seq_name = args.seq_name
    PDB_ID = args.pdb_id
    threshold = args.threshold
    method = args.method
    visualize_method = args.visualize_method
    use_raw_ig = args.use_raw_ig

    # load PDB data
    mol, ligand_name = get_mol(PDB_ID)

    l_epoch, l_pval, l_mean_val, l_acc = [], [], [], []
    for epoch_ in range(1, int(max_epoch)):
        try:
            epoch = str(epoch_).zfill(5)
#            print(f"# epoch = {epoch}")
            # =======================================
            # make save dir
            if visualize_method == 'ig':
                data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
                save_dir = f'viz/{dataset}/{model}/D_search_best_epoch/{seq_name}'
            else:
                pass
            os.makedirs(save_dir, exist_ok=True)

            # =======================================
            # load data
            data_seq, data_mol, data_igs = data_load(data_dir, seq_name, epoch)
            # IGのdata frameの作成
            df_igs_TP, df_igs_TN = get_df_IG(data_seq, data_mol, data_igs, use_raw_ig)

            # =======================================
            # important residuesの計算, タンパク質ごとにp-valueの平均値を計算
            dist_lr = DIST_LIGAND_RESIDUES()
            diff_resid_dict = dist_lr.residue_pick_mean(df_igs_TP, df_igs_TN, threshold)
            dist_t, dist_f, p_val, mean_val = dist_lr.dist_from_ligand(mol, diff_resid_dict, ligand_name)

            # =======================================
            # タンパク質ごとのACCを取得
            acc = data_seq['accuracy']

            # =======================================
            # append
            l_epoch.append(int(epoch))
            l_pval.append(p_val)
            l_mean_val.append(mean_val)
            l_acc.append(acc)

        except:
            pass

    # =======================================
    # 相関係数を計算
    df_1_ = pd.DataFrame(np.arange(int(max_epoch)), columns=['epoch'])
    df_2_ = pd.DataFrame([l_mean_val, l_acc, l_pval, l_epoch], columns=l_epoch, index=["diff_means", "ACC", "p-val", "epoch"]).T
    df = pd.merge(df_1_, df_2_, how="left", on='epoch')
#    corr = df['diff_means'].corr(df['ACC']) # 相関係数

    # best_epochを計算
    cal_best_epoch = CAL_BEST_EPOCH()
    TOP5_epoch, best_epoch = cal_best_epoch.method2(df)

    # epochごとのp-val,ACC,相関係数の推移
#    savepath = f'{save_dir}/scat_acc_dfmeans_{seq_name}_per{threshold}_method2.png'
#    fig_scat(df, corr, savepath)
    savepath = f'{save_dir}/line_acc_dfmeans_{seq_name}_per{threshold}.png'
    fig_line(df, best_epoch, seq_name, max_epoch, savepath)
    # save txt
    savepath = f'{save_dir}/acc_dfmeans_by_epoch_{seq_name}_per{threshold}.csv'
    df.to_csv(savepath)
    # save txt (TOP5 epoch)
    savepath = f'{save_dir}/top5_epoch_{seq_name}_per{threshold}.txt'
    f = open(savepath, 'w')
    for x in TOP5_epoch:
        f.write(str(x) + "\n")
    f.close()
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
    #
    parser.add_argument('--pdb_id', type=str, required=True,
                        help='pdb_id')
    # visualize method
    parser.add_argument('--visualize_method', default='ig', choices=['ig', 'grad_prod', 'grad', 'smooth_grad', 'smooth_ig'],
                        help='(ig, grad_prod, grad, smooth_grad, smooth_ig)')
    # important residues
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
    fd = open(f'{data_dir}/vizdata_{seq_name}_{epoch}.ckpt_seq.json', mode='r')
    data_seq = json.load(fd)
    fd.close()
    data_igs = joblib.load(f'{data_dir}/vizdata_{seq_name}_{epoch}.ckpt_igs.jbl')
    data_mol = joblib.load(f'{data_dir}/vizdata_{seq_name}_{epoch}.ckpt_mol.jbl')
    return data_seq, data_mol, data_igs

# %%


def get_mol(PDB_ID):
    from moleculekit.molecule import Molecule
    print('[INFO] get PDB data...')
    mol = Molecule(PDB_ID)
    mol.filter('chain A')
    # ligand name
    ligand_name = mol.get('resname', sel='not protein')[0]
    return mol, ligand_name

# %%


def get_df_IG(data_seq, data_mol, data_igs, use_raw_ig):
    # ====================
    # 対象の化合物を抽出 (条件 : pred_score > 0.7, TP, TN)
    def get_mol_list(data_mol):
        data_ = pd.DataFrame([data_mol['target_label'], data_mol['true_label'], data_mol['prediction_score']],
                             index=['target_label', 'true_label', 'prediction_score'], columns=data_mol['mol_id']).T
        data = data_[data_['prediction_score'] > 0.7]
        TP = data[(data['target_label'] == 1) & (data['true_label'] == 1)]
        TN = data[(data['target_label'] == 0) & (data['true_label'] == 0)]
#        print(f'# data count [TP={len(TP)}, TN={len(TN)}]')
        # extract mol index
        TP_mol_index = list(TP.index)
        TN_mol_index = list(TN.index)
        return TP_mol_index, TN_mol_index
    TP_mol_index, TN_mol_index = get_mol_list(data_mol)

    # ====================
    # IGのdfを作成
    if use_raw_ig == True:
        dtype = 'seq_IG'
    else:
        dtype = 'seq_IG_z'
    igs_TP_ = [data_igs[dtype][data_igs['mol_id'].index(p)] for p in TP_mol_index]
    df_igs_TP = pd.DataFrame(igs_TP_, index=TP_mol_index).iloc[:, :data_seq['seq_len']]
    igs_TN_ = [data_igs[dtype][data_igs['mol_id'].index(p)] for p in TN_mol_index]
    df_igs_TN = pd.DataFrame(igs_TN_, index=TN_mol_index).iloc[:, :data_seq['seq_len']]
    return df_igs_TP, df_igs_TN

# %%
class DIST_LIGAND_RESIDUES():
    def residue_pick_mean(self, df_igs_TP, df_igs_TN, threshold):
        seq_len = len(df_igs_TP.columns)
        # dfの平均のリストを抽出
        arr_TP_IG_mean = np.array(df_igs_TP.mean(axis=0))
        arr_TN_IG_mean = np.array(df_igs_TN.mean(axis=0))
        # 平均の差のランキング上位5%
        diff_IG_sub = np.subtract(arr_TP_IG_mean, arr_TN_IG_mean)
        # 平均値の差とTPの値の積→赤が強いところのみを抽出
        diff_IG_ = np.where(diff_IG_sub < 0, 0, diff_IG_sub)  # 0以上のみに抽出
        diff_IG = diff_IG_*arr_TP_IG_mean
        df_diff_IG = pd.DataFrame({'diff_IG': diff_IG})
        # TOPを取得
        #l_diff = np.array(df_diff_IG.sort_values("diff_IG", ascending=False).index)
        # diff_resid_dict = {resid: rank/10 for resid, rank in zip(l_diff, range(len(l_diff), 0, -1))}  # {残基番号:bfactor(順位降順)}
        # head(30)を取得
        head = int(seq_len*int(threshold)/100)
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
        coords_cas_t, coords_cas_f = get_coords_prot(
            mol, diff_resid_dict)  # 残基のCA座標
        coords_ligand = get_coords_ligand(mol, ligand_name)  # 対象のligandの重心座標
        # 距離のリスト
        from scipy.spatial import distance
        dist_t = [distance.euclidean(coords_ligand, coords_cas_t[i])
                  for i in list(coords_cas_t.keys())]
        dist_f = [distance.euclidean(coords_ligand, coords_cas_f[i])
                  for i in list(coords_cas_f.keys())]
        # 距離の平均値
        mean_dist_t = np.mean(dist_t)
        mean_dist_f = np.mean(dist_f)
        # 平均値の差 (対象残基の化合物距離の平均値 - 対象残基以外の化合物距離の平均値)
        mean_val = mean_dist_t - mean_dist_f
        # p-val
        from scipy import stats
        p_val = stats.ttest_ind(dist_t, dist_f, equal_var=True)[1]

        return dist_t, dist_f, p_val, mean_val


#%%
class CAL_BEST_EPOCH():
    # mean_diff上位5% * 前後5残基の平均値が高い
    def method1(self, df):
        # 50epoch以降を抽出
        df_over = df[df['epoch'] >= 50]
        # 最もmean_diffが良い（値が小さい）残基5%を抽出
        match_epoch_idx = sorted(list(df_over.sort_values('diff_means').head(int(len(df_over)*5/100))['epoch']))
        # best epochの計算
        #list_ar_epoch_mdiff_mean = []
        dict_ar_epoch_mdiff = {}
        for e in match_epoch_idx:
            # 抽出した残基の前後5個のmean_diffの平均値を計算
            df_ar_epoch_mdiff = df.query(f"epoch <= {int(e)+3} and epoch >= {int(e)-2}")
            ar_epoch_mdiff_mean = np.nanmean(np.array(df_ar_epoch_mdiff['diff_means']))  # 欠損を消して計算
            # list_ar_epoch_mdiff_mean.append(ar_epoch_mdiff_mean)
            dict_ar_epoch_mdiff[int(e)] = ar_epoch_mdiff_mean
        # 最もmdiffが良いepochを計算
        best_epoch = min(dict_ar_epoch_mdiff, key=dict_ar_epoch_mdiff.get)
        return best_epoch

    # ACCの順位+diff_meansの順位の和が小さい
    def method2(self, df):
        # cal ranking
        df["diff_means_rank"] = df.fillna(0)['diff_means'].rank()
        df["ACC_rank"] = df.fillna(0)['ACC'].rank(ascending=False)
        # ランキングが小さいepochを抽出
        df['rank_sum'] = df["diff_means_rank"] + df['ACC_rank']  # rankの和
        # TOP5
        best_epoch = int(df.sort_values('rank_sum', ascending=True)['epoch'].head(1))
        TOP5_epoch = list(df.sort_values('rank_sum', ascending=True)['epoch'].head(5).values.astype('int32'))
        print(TOP5_epoch)
        return TOP5_epoch, best_epoch


# %%
def fig_scat(df, corr, savepath):
    # fig ==================
    # config
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Pastel1')
    # plot
    fig, ax = plt.subplots(figsize=(12, 12))  # (親グラフ, 子グラフ)
    # kdeplot
    sns.scatterplot(data=df, x='diff_means', y='ACC', color="green")
    # lim
    #ax.set_xlim(-max(np.abs(df['diff_means'])), max(np.abs(df['diff_means'])))
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(-20, 20)
    # th
    ax.text(0.99, 0.95, f'corr = {round(corr, 3)}',
            va='top', ha='right', transform=ax.transAxes)
    # label
    #ax.set_xlabel('euclidean distance between ligand and residues [Å]')
    ax.invert_xaxis()
    # title
    #ax.title.set_text(f"Distribution of distance from ligand to TP/TN residues : {target}")
    # show
    plt.legend()
    # plt.show()
    # savefig
    fig.savefig(savepath)
    print(f'[SAVE] {savepath}')
    return

# %%


def fig_line(df, best_epoch, seq_name, max_epoch, savepath):
    # fig ==================
    # config
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Pastel1')
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    # plot
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    # kdeplot
    sns.barplot(data=df, ax=ax1, x="epoch", y="ACC", color="blue", alpha=0.3)
    sns.lineplot(data=df, ax=ax2, x="epoch", y="diff_means", color="red")
    # lim
    ax1.set_ylim((0.5, 1))
    #ax2.set_ylim(-max(np.abs(df['diff_means'])), max(np.abs(df['diff_means'])))
    ax2.set_ylim(-20, 20)
    ax1.set_xlim(0, int(max_epoch))
    ax2.set_xlim(0, int(max_epoch))
    # th
    #ax1.text(0.90, 0.95, f'corr = {round(corr, 3)}',va='top', ha='right', transform=ax2.transAxes)
    # label
    #ax.set_xlabel('euclidean distance between ligand and residues [Å]')
    ax1.axes.xaxis.set_visible(False)
    ax2.invert_yaxis()
    # background
    start_epoch = best_epoch - 2
    end_epoch = best_epoch + 2
    ax1.axvspan(start_epoch, end_epoch, color="gray", alpha=0.2)
    ax2.axvspan(start_epoch, end_epoch, color="gray", alpha=0.2)
    # title
    ax1.title.set_text(
        f"seq_name={seq_name}, best_epoch={best_epoch}, acc={np.round(df.at[best_epoch, 'ACC'], 3)}, p-val={np.round(df.at[best_epoch, 'p-val'], 3)}")
    # show
    plt.legend()
    # plt.show()
    # savefig
    fig.savefig(savepath)
    print(f'[SAVE] {savepath}')
    return


# %%
if __name__ == '__main__':
    main()
# %%


'''
    args = get_args()
    dataset = args.dataset
    model = args.model
    max_epoch = args.max_epoch
    seq_name = args.seq_name
    PDB_ID = args.pdb_id
    threshold = args.threshold
    method = args.method
    visualize_method = args.visualize_method
    use_raw_ig = args.use_raw_ig




'''
