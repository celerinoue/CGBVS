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
import itertools

# %%
def main():
    # args
    args = get_args()
    dataset = args.dataset
    model = args.model
    max_epoch = args.max_epoch
    seq_name = args.seq_name
    #PDB_ID = args.pdb_id
    threshold = args.threshold
    method = args.method
    #visualize_method = args.visualize_method
    use_raw_ig = args.use_raw_ig
    plif_result_path = args.plif_result

##====================================
    # load plif data
    plif_result, plif_res_list = load_plif_result(plif_result_path, seq_name)
    PDB_ID = plif_result['pdb_name'].mode()[0]
    print(f'# PDB_ID = {PDB_ID}')
    # load PDB data
    mol, ligand_name = get_mol(PDB_ID)

##====================================
    # domain res index data
    if "domain" in dataset:
        domain_location = load_domain_location(f"data/other/domain_location/kinase_chembl_domain.csv", seq_name)
    else:
        print('# no domain data')

    l_epoch, l_acc, l_ratio_0A, l_ratio_3A, l_ratio_5A, l_ratio_7A, l_ratio_10A, l_ratio_15A = [],[],[],[],[],[],[],[]
    for epoch_ in range(1, int(max_epoch)):
        try:
            epoch = str(epoch_).zfill(5)
            print(f"# epoch = {epoch}")
            # =======================================
            # make save dir
            data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
            save_dir = f'viz/{dataset}/{model}/D_search_best_epoch_ratio_res_plif/{seq_name}/'
            os.makedirs(save_dir, exist_ok=True)
            # =======================================
            # load data
            data_seq, data_mol, data_igs = data_load(data_dir, seq_name, epoch)
            # IGのdata frameの作成
            df_igs_TP, df_igs_TN = get_df_IG(data_seq, data_mol, data_igs, use_raw_ig)
            # important resid [list_hit_resid = uniprot残基番号を出力]
            list_hit_resid = get_hit_resid(df_igs_TP, df_igs_TN, domain_location, threshold)
            print(list_hit_resid)
            # =======================================
            # ↓ここから立体構造(PLIF)を使う
            # important residuesの計算, タンパク質ごとにp-valueの平均値を計算
            ratio_0A = cal_ratio_plif(list_hit_resid, plif_res_list, mol, 0)
            ratio_3A = cal_ratio_plif(list_hit_resid, plif_res_list, mol, 3)
            ratio_5A = cal_ratio_plif(list_hit_resid, plif_res_list, mol, 5)
            ratio_7A = cal_ratio_plif(list_hit_resid, plif_res_list, mol, 7)
            ratio_10A = cal_ratio_plif(list_hit_resid, plif_res_list, mol, 10)
            ratio_15A = cal_ratio_plif(list_hit_resid, plif_res_list, mol, 15)
            # =======================================
            # タンパク質ごとのACCを取得
            acc = data_seq['accuracy']
            # =======================================
            # append
            l_epoch.append(int(epoch))
            l_acc.append(acc)
            l_ratio_0A.append(ratio_0A)
            l_ratio_3A.append(ratio_3A)
            l_ratio_5A.append(ratio_5A)
            l_ratio_7A.append(ratio_7A)
            l_ratio_10A.append(ratio_10A)
            l_ratio_15A.append(ratio_15A)
        except:
            #print("[PASS]")
            pass

    # =======================================
    # 相関係数を計算
    df_epoch_ = pd.DataFrame(np.arange(int(max_epoch)), columns=['epoch'])
    df_ratio_ = pd.DataFrame([l_epoch, l_acc, l_ratio_0A, l_ratio_3A, l_ratio_5A, l_ratio_7A, l_ratio_10A, l_ratio_15A],
                                columns=l_epoch,
                                index=["epoch", "ACC", "ratio_0A", "ratio_3A", "ratio_5A", "ratio_7A", "ratio_10A", "ratio_15A"]).T
    df = pd.merge(df_epoch_, df_ratio_, how="left", on='epoch')
    savepath = f'{save_dir}/acc_plif_by_epoch_{seq_name}_per{threshold}.csv'
    df.to_csv(savepath)

    # best_epochを計算
    cal_best_epoch = CAL_BEST_EPOCH()
    round_list = [0,3,5,7,10,15]
    for round in round_list:
        TOP5_epoch, best_epoch = cal_best_epoch.method2(df, round)
        # epochごとのACC, 可視化結果(PLIF)の推移
        savepath = f'{save_dir}/line_acc_dfmeans_{seq_name}_per{threshold}_round{round}.png'
        fig_line(df, round, best_epoch, seq_name, max_epoch, savepath)
        # save txt (TOP5 epoch)
        savepath = f'{save_dir}/top5_epoch_{seq_name}_per{threshold}_round{round}.txt'
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
    parser.add_argument('--pdb_id', type=str, required=False,
                        help='pdb_id')
    parser.add_argument('--plif_result', type=str, required=True,
                        help='plif_result_dir')
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


#%%
def load_plif_result(plif_result_path, seq_name):
    plif_result_ = pd.read_csv(plif_result_path, index_col=0)
    plif_result = plif_result_[plif_result_['uniprot_name'] == seq_name]
    plif_res_list = sorted(list(plif_result['resnum_uniprot'].unique()))
    return plif_result, plif_res_list


# %%
def get_mol(PDB_ID):
    from moleculekit.molecule import Molecule
    print('[INFO] get PDB data...')
    mol = Molecule(PDB_ID)
    mol.filter('chain A')
    # ligand name
    ligand_name = mol.get('resname', sel='not protein')[0]
    return mol, ligand_name

#%%
def load_domain_location(path, seq_name):
    df1 = pd.read_csv(path)
    df2 = df1[df1['uniProtkbId'] == seq_name]
    domain_location = {"start":int(df2['start']), "end":int(df2['end'])}
    return domain_location

#%%
def convert_resid_index_to_uniprot(resid_index, start_resid):
    # domain切り出しindexからuniprotの残基番号に戻す
    if type(resid_index) == int:
        uniprot_resid = int(resid_index + start_resid)
    elif type(resid_index) == list:
        uniprot_resid = sorted([int(int(r) + start_resid) for r in resid_index])
    else:
        print("[ERROR] type() must int or list")
    return uniprot_resid

# %%
def get_df_IG(data_seq, data_mol, data_igs, use_raw_ig):
    # ====================
    # 対象の化合物を抽出 (条件 : pred_score > 0.7, TP, TN)
    def get_mol_list(data_mol):
        data_ = pd.DataFrame([data_mol['target_label'], data_mol['true_label'], data_mol['prediction_score']],
                             index=['target_label', 'true_label', 'prediction_score'], columns=data_mol['mol_id']).T
        data = data_[data_['prediction_score'] > 0]
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

#%%
def get_hit_resid(df_igs_TP, df_igs_TN, domain_location, threshold):
    seq_len = len(df_igs_TP.columns)
    res_count = int(seq_len*int(threshold)/100)  # 抽出する残基の数
    # dfの平均のリストを抽出
    arr_TP_IG_mean = np.array(df_igs_TP.mean(axis=0))
    arr_TN_IG_mean = np.array(df_igs_TN.mean(axis=0))
    print(f"TP/TN {len(df_igs_TP), len(df_igs_TN)}")
    diff_IG_sub = np.subtract(arr_TP_IG_mean, arr_TN_IG_mean) # 平均の差
    df_diff_IG = pd.DataFrame({'diff_IG': diff_IG_sub})
    # index取得
    list_hit_resid_index = np.array(df_diff_IG.sort_values("diff_IG", ascending=False).head(res_count).index)
    # index番号から uniprotの残基番号に変換
    list_hit_resid_uniprot = convert_resid_index_to_uniprot(list(list_hit_resid_index), domain_location['start'])
    return list_hit_resid_uniprot

# %%
def cal_ratio_plif(list_hit_resid, plif_res_list, mol, round):
    # get contact_map (round xxx)
    def coords_map(mol):
        from scipy.spatial.distance import cdist
        resnum = mol.get('resid','name CA') # kinaseドメインではない！！PDBのresid
        coords = mol.get('coords','name CA')
        distance_map_ = cdist(coords, coords, metric="euclidean")
        distance_map = pd.DataFrame(distance_map_, index=resnum, columns=resnum)
        return distance_map, resnum
    distance_map, resnum = coords_map(mol)
    contact_map = (distance_map <= round).astype(int)
    # get index
    def arr_index(l, x):
        return [i for i, _x in enumerate(list(l)) if _x == x]
    contact_resids = set()
    for res in plif_res_list:
        try:
            index_ = arr_index(contact_map[res], 1)
            contact_resids_ = [contact_map.columns[i] for i in index_]
            contact_resids = contact_resids | set(contact_resids_)
        except:
            pass
    # 割合 PLIFに限定してどれくらい入っているか
    ratio = len(set(list_hit_resid) & set(contact_resids)) / len(list_hit_resid)
    return ratio

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
    def method2(self, df, round):
        col_ = f"ratio_{round}A"
        # cal ranking
        df["vis_rank"] = df.fillna(0)[col_].rank(ascending=False)
        df["ACC_rank"] = df.fillna(0)['ACC'].rank(ascending=False)
        # ランキングが小さいepochを抽出
        df['rank_sum'] = df["vis_rank"] + df['ACC_rank']  # rankの和
        # TOP5
        best_epoch = int(df.sort_values('rank_sum', ascending=True)['epoch'].head(1))
        TOP5_epoch = list(df.sort_values('rank_sum', ascending=True)['epoch'].head(5).values.astype('int32'))
        print(TOP5_epoch)
        return TOP5_epoch, best_epoch

# %%
def fig_line(df, round, best_epoch, seq_name, max_epoch, savepath):
    col_ = f"ratio_{round}A"
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
    sns.lineplot(data=df, ax=ax2, x="epoch", y=col_, color="red")
    # lim
    ax1.set_ylim((0.5, 1))
    #ax2.set_ylim(-max(np.abs(df['diff_means'])), max(np.abs(df['diff_means'])))
    ax2.set_ylim(0, 1)
    ax1.set_xlim(0, int(max_epoch))
    ax2.set_xlim(0, int(max_epoch))
    # th
    #ax1.text(0.90, 0.95, f'corr = {round(corr, 3)}',va='top', ha='right', transform=ax2.transAxes)
    # label
    #ax.set_xlabel('euclidean distance between ligand and residues [Å]')
    ax1.axes.xaxis.set_visible(False)
    #ax2.invert_yaxis()
    # background
    start_epoch = best_epoch - 2
    end_epoch = best_epoch + 2
    ax1.axvspan(start_epoch, end_epoch, color="gray", alpha=0.2)
    ax2.axvspan(start_epoch, end_epoch, color="gray", alpha=0.2)
    # title
    ax1.title.set_text(f"seq_name={seq_name}, best_epoch={best_epoch}, acc={np.round(df.at[best_epoch, 'ACC'], 3)}")
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
dataset = "kinase_chembl_domain_small_test"
model = "model_8_2"
max_epoch = "500"
seq_name = "LCK_HUMAN"
threshold = "5"
method = "mdiff"
visualize_method = "ig"
use_raw_ig = False
plif_result_path = "data/other/PLIF/analyzed_kinase_MOE_result2/plif_result.csv"
'''
