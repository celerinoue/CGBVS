# Author: S.Inoue
# Date: 07/24/2022
# Updated: 07/24/2022


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
import sys
from statistics import mode
from tqdm import tqdm

# %%
def main():
    ## args
    args = get_args()
    ## ===== main data & config ======
    dataset = args.dataset
    model = args.model
    seq_name = args.seq_name
    max_epoch = args.max_epoch
    plif_result_path = args.plif_result
    ## ===== method ======
    score_cal_method = args.score_cal_method # [plif_m1~3, cmol_m1, ...]
    ratio_ig_res = int(args.ratio_ig_res)  # ratio of detected residues []
    #cutoff_roundA = int(args.cutoff_roundA)  # cutoff (in Å) [0,1,3,5,8,10,15 Å]
    epoch_search_method = args.epoch_search_method
    ## ===== other option ======
    use_raw_ig = args.use_raw_ig

    ##====================================
    ## load & check PLIF data
    plif_result, plif_res_list = load_plif_result(plif_result_path, seq_name) # load plif data
    print(f'# PLIF_COUNT = {len(plif_res_list)}')
    if plif_result is not None:
        PDB_ID = plif_result['pdb_name'].mode()[0] # 最もPLIFデータが多い構造を鋳型に用いる
        ## load PDB data
        mol, ligand_name = get_mol(PDB_ID)
        print(f'# PDB_ID = {PDB_ID}')
        print(f'# ligand = {ligand_name}')
    else:
        print("[INFO] plif data is None")
        sys.exit()

    ## load domain res index data
    if "domain" in dataset:
        domain_location = load_domain_location(f"data/other/domain_location/kinase_chembl_domain.csv", seq_name)
        print(f'# kinase domain location = {domain_location}')
    else:
        print('[ERROR] no domain data')
        sys.exit()

    ## make save dir
    data_dir = f'viz/{dataset}/{model}/A_process/{seq_name}'
    save_dir = f'viz/{dataset}/{model}/D_search_best_epoch/{score_cal_method}/{seq_name}'
    os.makedirs(save_dir, exist_ok=True)

    ##====================================
    ## load IG data
    print("[INFO] load IG data")
    l_epoch, l_ig_detected_resid, l_score_acc = [],[],[]
    for epoch_ in range(1, int(max_epoch)):
        try:
            epoch = str(epoch_).zfill(5)
            ##=======================================
            ## load data
            data_seq, data_mol, data_igs = data_load(data_dir, seq_name, epoch)
            ## IGのdata frameの作成
            df_igs_TP, df_igs_TN = get_df_IG(data_seq, data_mol, data_igs, use_raw_ig)
            print(f"# TP={len(df_igs_TP)}, TN={len(df_igs_TN)} (epoch = {epoch})")
            ## ig_detected_resid_list,  (uniprot残基番号)
            ig_detected_resid = get_hit_resid(df_igs_TP, df_igs_TN, domain_location, ratio_ig_res)
            ##=======================================
            ## acc per epoch
            score_acc = data_seq['accuracy']
            ##=======================================
            ## append
            l_epoch.append(epoch)
            l_ig_detected_resid.append(ig_detected_resid)
            l_score_acc.append(score_acc)
        except:
            print(f"[PASS] IG data is None (epoch = {epoch})")
            pass

    ##====================================
    cal_score = CAL_SCORE_IG_VIZ()
    round_list = [0,4,5,6,8,10,15] ############################################
    for cutoff_roundA in tqdm(round_list):
        ## annotate residues (round in ?Å) & cal ratio
        distance_map = get_distance_map(mol)
        l_score_ig_viz = []
        for i in range(len(l_epoch)):
            if score_cal_method == "plif_m1":
                score_ig_viz = cal_score.plif_m1(l_ig_detected_resid[i], distance_map, plif_res_list, cutoff_roundA)
            elif score_cal_method == "plif_m2":
                score_ig_viz = cal_score.plif_m2(l_ig_detected_resid[i], distance_map, plif_res_list, cutoff_roundA)
            elif score_cal_method == "plif_m3":
                score_ig_viz = cal_score.plif_m3(l_ig_detected_resid[i], distance_map, plif_res_list, cutoff_roundA)
            elif score_cal_method == "cmol_m1":
                score_ig_viz = cal_score.cmol_m1(l_ig_detected_resid[i], cutoff_roundA, mol, ligand_name)
            ## append
            l_score_ig_viz.append(score_ig_viz)

        ##====================================
        df_epoch_ = pd.DataFrame(np.arange(int(max_epoch)), columns=['epoch'])
        df_ratio_ = pd.DataFrame([[int(n) for n in l_epoch], l_score_acc, l_score_ig_viz],
                                    columns=l_epoch,
                                    index=["epoch", "score_ACC", f"score_viz"]).T
        df = pd.merge(df_epoch_, df_ratio_, how="left", on='epoch')
        savepath = f'{save_dir}/score_{score_cal_method}_per{ratio_ig_res}_round{cutoff_roundA}.csv'
        df.to_csv(savepath)

        # best_epochを計算
        cal_best_epoch = CAL_BEST_EPOCH()
        if epoch_search_method == 'method2':
            best_epoch, epoch_rank_all = cal_best_epoch.method2(df)
        elif epoch_search_method == 'method3':
             best_epoch, epoch_rank_all = cal_best_epoch.method3(df)
        # epochごとのACC, 可視化結果(PLIF)の推移
        savepath = f'{save_dir}/epochsearch_{epoch_search_method}_{score_cal_method}_per{ratio_ig_res}_round{cutoff_roundA}.png'
        fig_line(df, cutoff_roundA, best_epoch, seq_name, max_epoch, savepath)
        # save txt (TOP5 epoch)
        savepath = f'{save_dir}/epochsearch_{epoch_search_method}_{score_cal_method}_per{ratio_ig_res}_round{cutoff_roundA}.txt'
        f = open(savepath, 'w')
        for x in epoch_rank_all:
            f.write(str(x) + "\n")
        f.close()

        ##====================================
        ## write pdb
        ## cal residues
        #ig_residues = cal_contact_resids(distance_map, cutoff_roundA, l_ig_detected_resid[best_epoch])
        plif_residues = cal_contact_resids(distance_map, cutoff_roundA, plif_res_list)
        ## save pdb
        savepath = f'{save_dir}/map_ig_detected_residues_per{ratio_ig_res}_epoch{str(best_epoch)}.pdb'
        write_pdb(mol, l_ig_detected_resid[best_epoch], savepath) # score_viz
        savepath = f'{save_dir}/map_plif_residues_round{cutoff_roundA}.pdb'
        write_pdb(mol, plif_residues, savepath)  # score_plif

    return


# %%
def get_args():
    parser = argparse.ArgumentParser(description='sample text', add_help=True)
    ## ===== main data & config ======
    parser.add_argument('--dataset', type=str, required=True,
                        help='')
    parser.add_argument('--model', type=str, required=True,
                        help='')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='(e.g. --seq_name LCK_HUMAN)')
    parser.add_argument('--max_epoch', type=str, required=True,
                        help='')
    parser.add_argument('--plif_result', type=str, required=True,
                        help='plif_result_dir')
    ## ===== method ======
    parser.add_argument('--score_cal_method', choices=['plif_m1', 'plif_m2', 'plif_m3', 'cmol_m1'], default='plif_m1',
                        help='chooose method [plif_m1~m3, cmol_m1]')
    parser.add_argument('--ratio_ig_res', type=str, default='5',
                        help='(e.g. --ratio_ig_res 0,5,10)')
    parser.add_argument('--cutoff_roundA', type=str, default='5',
                        help='(e.g. --cutoff_roundA 0,5,10)')
    parser.add_argument('--epoch_search_method', choices=['method2', 'method3'], default='method2',
                        help='chooose method [method2, method3]')
    ## ===== other option ======
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
    plif_result_ = plif_result_[plif_result_['uniprot_name'] == seq_name]
    # 一部の化合物を除外
    plif_result = plif_result_.query('ligand_name != [" MG", "TPO", "EDO", "SO4", "PO4", "CSO", "ACT", "GOL"]')
    plif_res_list = sorted(list(plif_result['resnum_uniprot'].unique()))
    return plif_result, plif_res_list


# %%
def get_mol(PDB_ID):
    from moleculekit.molecule import Molecule
    print('[INFO] get PDB data...')
    mol = Molecule(PDB_ID)
    mol.filter('chain A')
    # ligand name
    ligand_name = mode([s for s in mol.get('resname', sel='not protein') if s != 'HOH'])
    # 水分子[HOH]以外で最も出現回数が多いものをリガンドとした
    return mol, ligand_name

def get_distance_map(mol):
    from scipy.spatial.distance import cdist
    resnum = mol.get('resid', sel='name CA and protein') # kinaseドメインではない！！PDBのresid
    coords = mol.get('coords', sel='name CA and protein')
    distance_map_ = cdist(coords, coords, metric="euclidean")
    distance_map = pd.DataFrame(distance_map_, index=resnum, columns=resnum)
    return distance_map

def cal_contact_resids(distance_map, cutoff_roundA, target_resid):
    ## get contact_map
    contact_map = (distance_map <= cutoff_roundA).astype(int)
    ## get contact resids
    def arr_index(l, x):  # contactmapのindexを取得
        return [i for i, _x in enumerate(list(l)) if _x == x]
    contact_resids = set()
    for res in target_resid:
        if res in contact_map.index:
            index_ = arr_index(contact_map[res], 1)
            contact_resids_ = [contact_map.columns[i] for i in index_]
            contact_resids = contact_resids | set(contact_resids_)
        else:
            pass
    return contact_resids


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
def get_hit_resid(df_igs_TP, df_igs_TN, domain_location, ratio_ig_res):
    seq_len = len(df_igs_TP.columns)
    res_count = int(seq_len*int(ratio_ig_res)/100)  # 抽出する残基の数
    # dfの平均のリストを抽出
    arr_TP_IG_mean = np.array(df_igs_TP.mean(axis=0))
    arr_TN_IG_mean = np.array(df_igs_TN.mean(axis=0))
    diff_IG_sub = np.subtract(arr_TP_IG_mean, arr_TN_IG_mean) # 平均の差
    df_diff_IG = pd.DataFrame({'diff_IG': diff_IG_sub})
    # index取得
    hit_resid_list_index = np.array(df_diff_IG.sort_values("diff_IG", ascending=False).head(res_count).index)
    # index番号から uniprotの残基番号に変換
    hit_resid_list_uniprot = convert_resid_index_to_uniprot(list(hit_resid_list_index), domain_location['start'])
    return hit_resid_list_uniprot

# %%
class CAL_SCORE_IG_VIZ():  ## ratio calculation method
    def plif_m1(self, ig_detected_resid, distance_map, plif_res_list, cutoff_roundA):
        ## ratio=(IG残基の周囲?Å以内に存在するPLIF残基の数)/(PLIF残基の総数)
        contact_resids = cal_contact_resids(distance_map, cutoff_roundA, ig_detected_resid)
        ## top / bottom
        numer = len(set(plif_res_list) & set(contact_resids)) #
        denom = len(set(plif_res_list) & set(distance_map.index)) # PDBの残基IDに含まれるPLIF残基
        return numer / denom

    def plif_m2(self, ig_detected_resid, distance_map, plif_res_list, cutoff_roundA):
        ## ratio=(IG残基の周囲?Å以内にPLIF残基を1つでも含むIG残基の数)/(IG残基の総数)
        res_count = 0
        for r in ig_detected_resid:
            if len(set(plif_res_list) & set(cal_contact_resids(distance_map, cutoff_roundA, [r]))) >= 1:
                res_count += 1
        ## top / bottom
        numer = res_count
        denom = len(set(ig_detected_resid) & set(distance_map.index)) # PDBの残基IDに含まれるPLIF残基
        return numer / denom

    def plif_m3(self, ig_detected_resid, distance_map, plif_res_list, cutoff_roundA):
        ## ratio=(PLIFの周囲?Å以内に存在するIG残基の数)/(IG残基の総数) [× old]
        contact_resids = cal_contact_resids(distance_map, cutoff_roundA, plif_res_list)
        return len(set(ig_detected_resid) & set(contact_resids)) / len(ig_detected_resid)

    def cmol_m1(self, ig_detected_resid, cutoff_roundA, mol, ligand_name):
        ## (化合物原子の全座標から周囲?Å以内に存在するIG残基の数)/(IG残基の総数)
        ## 化合物原子とタンパク質残基とのdistance_map
        from scipy.spatial.distance import cdist
        def get_coords_resids(mol, ligand_name):
            resnum = mol.get('resid', sel='name CA and protein') # kinaseドメインではない！！PDBのresid
            coords_resids = mol.get('coords', sel='name CA and protein')
            atom_names = mol.get('name', sel=f'resname {ligand_name}')
            coords_ligands = mol.get('coords', f'resname {ligand_name}')
            return coords_resids, coords_ligands, resnum, atom_names
        coords_resids, coords_ligands, resnum, atom_names = get_coords_resids(mol, ligand_name)
        distance_map_ = cdist(coords_resids, coords_ligands, metric="euclidean")
        distance_map = pd.DataFrame(distance_map_, index=resnum, columns=atom_names)
        ## pocket resid
        pocket_resids = []
        for i in distance_map.index:
            if np.array(distance_map.loc[i]).min() <= cutoff_roundA:
                pocket_resids.append(i)
        ## top / bottom
        numer = len(set(ig_detected_resid) & set(pocket_resids))
        denom = len(set(ig_detected_resid) & set(distance_map.index)) # PDBの残基IDに含まれるPLIF残基
        return numer / denom


#%%
class CAL_BEST_EPOCH():
    '''
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
    '''

    # ACCの順位+diff_meansの順位の和が小さい
    def method2(self, df):
        # cal ranking
        df["viz_rank"] = df.fillna(0)['score_viz'].rank(ascending=False)
        df["ACC_rank"] = df.fillna(0)['score_ACC'].rank(ascending=False)
        # ランキングが小さいepochを抽出
        df['rank_sum'] = df["viz_rank"] + df['ACC_rank']  # rankの和
        # TOP5
        best_epoch = int(df.sort_values('rank_sum', ascending=True)['epoch'].head(1))
        epoch_rank_all = list(df.sort_values('rank_sum', ascending=True)['epoch'].values.astype('int32'))
        return best_epoch, epoch_rank_all


# %%
def fig_line(df, cutoff_roundA, best_epoch, seq_name, max_epoch, savepath):
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
    sns.barplot(data=df, ax=ax1, x="epoch", y="score_ACC", color="blue", alpha=0.3)
    sns.lineplot(data=df, ax=ax2, x="epoch", y="score_viz", color="red")
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
    ax1.title.set_text(f"seq={seq_name}, round={cutoff_roundA}Å, best_epoch={best_epoch}, acc={np.round(df.at[best_epoch, 'score_ACC'], 3)}, score_viz={np.round(df.at[best_epoch, 'score_viz'], 3)}")
    # show
    plt.legend()
    # plt.show()
    # savefig
    fig.savefig(savepath)
    print(f'[SAVE] {savepath}')
    return

#%%
##===================================
def write_pdb(mol, target_resids, savepath):
    ## original PDB data
    resnum = mol.get('resid', sel='name CA and protein')
    viz_result_dict = {i: 0 for i in resnum}
    for i in target_resids:
        viz_result_dict[i] = 1
    ## get bfactor_arr
    bfactor_arr = np.array([viz_result_dict.get(i) for i in mol.get('resid', sel='protein')])
    ## embedding & save data
    mol.set("beta", bfactor_arr, sel='protein')
    mol.write(savepath, sel='protein')
    return


# %%
if __name__ == '__main__':
    main()
# %%




'''
# config
dataset = "kinase_chembl_domain_small_test"
model = "model_8_2"
max_epoch = "500"
seq_name = "CDK2_HUMAN"
plif_result_path = "data/other/PLIF/analyzed_kinase_MOE_result2/plif_result.csv"
score_cal_method = 'plif_m1'
ratio_ig_res = 5
cutoff_roundA = 5
epoch_search_method = 'method2'
use_raw_ig = False
'''
