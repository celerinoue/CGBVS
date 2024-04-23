
#%%
#from lib2to3.pgen2.pgen import DFAState
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import glob
import pathlib
from rdkit import Chem
import shutil
#from moleculekit.molecule import Molecule
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


#%%
def main():
    ## config
    th = 30 # μM
    datatype = "sample-set" # [refined or general]

    ## load data
    #df_name = load_pl_name(f'data/original/PDBbind/index/INDEX_{datatype}_PL_name.2020')
    #df_data_r = load_pl_data(f'data/original/PDBbind/index/INDEX_{datatype}_PL_data.2020')
    df_name = load_pl_name(f'data/original/PDBbind/index/INDEX_{datatype}_name.2020')
    df_data_r = load_pl_data(f'data/original/PDBbind/index/INDEX_{datatype}_data.2020')

    ## general set
    # List of protein-ligand complexes with known binding data in PDBbind v.2020
    # 19443 complexes in total, clustered by 95% protein sequence similarity

    ## refined set
    # List of protein-ligand complexes in the PDBbind refined set v.2020
    # 5316 complexes in total, clustered by 90% protein sequence similarity

    ## data cleaning
    df_data_rc = data_clean(df_data_r, th)

    ## label
    df_data = get_label(data=df_data_rc, th=th) # μM

    ## merge
    df = pd.merge(df_data, df_name, on='pdb_id', how='left')

    ## extract only kinase
    uniprot_name_kinase = pd.read_csv('data/other/uniprot/uniprot-query_kinase_human.tsv', sep='\t')
    df_kinase = pd.merge(df, uniprot_name_kinase, left_on="Uniprot ID", right_on="Entry", how="left").dropna(how='any')
    print(f'[INFO] data count={len(df_kinase)}, active={len(df_kinase[df_kinase["label"]==1])}, inactive={len(df_kinase[df_kinase["label"]==0])}')
    #[["pdb_id", "reference", "ligand_name", "label", "Uniprot ID", "Entry Name"]]

    ## save data ================
    os.makedirs(f'data/original/kinase_PDBbind_{datatype}', exist_ok=True)
    df_kinase.to_csv(f'data/original/kinase_PDBbind_{datatype}/df_all_kinase.csv')

    save_data(df_kinase, datatype)

    return


#%%
def load_pl_name(path):
    ## read file
    f = open(path, 'r', encoding='UTF-8')
    list = [s for s in f.readlines() if not s.startswith('#')]
    pdb_id = [l[:4] for l in list]
    #release_year = [l[4:10].strip() for l in list]
    uniprot_id = [l[12:20].strip() for l in list]
    protein_name = [l[21:].strip() for l in list]
    f.close()
    pl_name = pd.DataFrame([pdb_id, uniprot_id, protein_name],
                            index=['pdb_id', 'Uniprot ID', 'protein name']).T
    return pl_name


#%%
def load_pl_data(path):
    ## read file ===========-
    f = open(path, 'r', encoding='UTF-8')
    list = [s for s in f.readlines() if not s.startswith('#')]
    pdb_id = [l[:4] for l in list]
    resolution = [l[4:10].strip() for l in list]
    release_year = [l[11:16].strip() for l in list]
    log_kdki = [l[17:23].strip() for l in list]
    kdki = [l[24:39].strip() for l in list]
    reference = [l[42:50].strip() for l in list]
    ligand_name = [l[51:].strip("( ) \n") for l in list]
    f.close()

    ## data reshape =========
    ## col kd,ki,IC50
    label, rel, af, c = [],[],[],[]
    for d in kdki:
        if '<=' in d:
            label.append(d.split('<=')[0]) # Ki
            rel.append('<=')  # =, <
            af.append(d.split('<=')[1][:-2]) # 100
            c.append(d.split('<=')[1][-2:]) # nM
        elif '>=' in d:
            label.append(d.split('>=')[0])  # Ki
            rel.append('>=')  # =, <
            af.append(d.split('>=')[1][:-2])  # 100
            c.append(d.split('>=')[1][-2:])  # nM
        elif '~' in d:
            label.append(d.split('~')[0])  # Ki
            rel.append('~')  # =, <
            af.append(d.split('~')[1][:-2])  # 100
            c.append(d.split('~')[1][-2:])  # nM
        elif '>' in d:
            label.append(d.split('>')[0])  # Ki
            rel.append('>')  # =, <
            af.append(d.split('>')[1][:-2])  # 100
            c.append(d.split('>')[1][-2:])  # nM
        elif '<' in d:
            label.append(d.split('<')[0])  # Ki
            rel.append('<')  # =, <
            af.append(d.split('<')[1][:-2])  # 100
            c.append(d.split('<')[1][-2:])  # nM
        elif '=' in d:
            label.append(d.split('=')[0])  # Ki
            rel.append('=')  # =, <
            af.append(d.split('=')[1][:-2])  # 100
            c.append(d.split('=')[1][-2:])  # nM
        else:
            print("[ERROR]")

    ## set dataframe
    pl_data = pd.DataFrame([pdb_id,resolution,release_year,log_kdki,label,rel,af,c,reference,ligand_name],
                      index=['pdb_id', 'resolution', 'release_year', 'log_kdki', 'af_kikd', 'af_rel','af_val','af_unit','reference','ligand_name']).T

    ## unit
    vals = [] # μMに揃える
    for i,unit in enumerate(pl_data['af_unit'].tolist()):
        if unit == 'mM':
            val_ = float(pl_data['af_val'][i])*1000
        elif unit == 'uM':
            val_ = float(pl_data['af_val'][i])*1
        elif unit == 'nM':
            val_ = float(pl_data['af_val'][i])/1000
        elif unit == 'pM':
            val_ = float(pl_data['af_val'][i])/1000000
        elif unit == 'fM':
            val_ = float(pl_data['af_val'][i])/1000000000
        else:
            print('[ERROR]')
            pass
        vals.append(val_)
    pl_data['af_val'] = vals
    pl_data['af_unit'] = 'μM'
    return pl_data


#%%
def data_clean(data_, th):
    a = data_[data_["af_rel"] == "="]
    b = data_[(data_['af_rel'] == '<') & (data_['af_val'] <= th)]  # n=149 30μM以上で削除
    c = data_[(data_['af_rel'] == '<=') & (data_['af_val'] <= th)] # n=7
    d = data_[data_["af_rel"] == "~"]
    e = data_[(data_['af_rel'] == '>') & (data_['af_val'] > th)] # n=215 30μM以下で削除
    f = data_[(data_['af_rel'] == '>=') & (data_['af_val'] > th)] # n=1 削除
    ## merge
    data = pd.concat([a,b,c,d,e,f])
    return data


#%%
def get_label(data, th=30):
    ## label
    label = []
    for val in data['af_val'].tolist():
        if val < th:
            label.append(1) # active
        elif val >= th:
            label.append(-1) # inactive

    ## add
    data['label'] = label

    return data



# %%
def save_data(df_kinase, datatype):

    for i in range(len(df_kinase)):
        ## data
        pdb_id = df_kinase['pdb_id'].iloc[i]
        #ligand_name = df_kinase['ligand_name'][12]
        ligand_name = f'{pdb_id}_ligand'
        label = df_kinase['label'].iloc[i]
        print(pdb_id)

        ## dir SDF_wash, SDF
        data_dir = f'data/original/kinase_PDBbind_{datatype}/{pdb_id}'
        os.makedirs(f"{data_dir}/SDF_wash", exist_ok=True)
        os.makedirs(f"{data_dir}/SDF", exist_ok=True)

        try:
            ## mv sdf & sdf_wash
            sdf_data_path = f'data/other/PDBbind/{datatype}-set/{pdb_id}/{pdb_id}_ligand.sdf'
            shutil.copyfile(sdf_data_path, f'{data_dir}/SDF/SDF.sdf')  # cp
            #sdf_data_path = f'data/other/PDBbind/{datatype}-set/{pdb_id}/{pdb_id}_ligand_wash.sdf'
            #shutil.copyfile(sdf_data_path, f'{data_dir}/SDF_wash/SDF_wash.sdf') # cp
        except:
            print(f"[ERROR] {pdb_id} no SDF data")

        ## assay csv
        pd.DataFrame([ligand_name, label]).T.to_csv(f"{data_dir}/assay.csv", header=False, index=False, sep='\t')



        ## interpro.txt
        p = pathlib.Path(f"{data_dir}/interpro.txt")
        with p.open(mode='w') as f:
            f.write('')

    return




# %%
if __name__ == '__main__':
    main()
# %%
