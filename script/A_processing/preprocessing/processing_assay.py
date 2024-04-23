#%%
#from lib2to3.pgen2.pgen import DFAState
import pandas as pd
import numpy as np
import sys
import re
import os
import glob
import pathlib
from rdkit import Chem
import shutil
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import *

#%%
def main():
    ## config & args
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    label_th = 30 # μM
    assay_index_name_path = sys.argv[3]
    assay_index_data_path = sys.argv[4]
    uniprot_kinase_list_path = '/data_st02/drug/inoue/CGBVS/data/other/uniprot/uniprot-query_kinase-protein-superfamily.tsv'

    ##============================
    ## load assay result data
    data_list = [d.split('/')[-1] for d in glob.glob(f'{data_dir}/*')]
    print(f'{len(data_list)} data found')
    savepath = f'{save_dir}/dataset_reference.csv'
    df_kinase = processing_reference(data_list, assay_index_name_path, assay_index_data_path, uniprot_kinase_list_path, label_th, savepath)

    pdb_ids = list(df_kinase['pdb_id'])

    ##============================
    ## make directory
    er = []
    for pdb_id in pdb_ids:
        os.makedirs(f'{save_dir}/{pdb_id}/SDF_wash/', exist_ok=True) ## make dir
        data = df_kinase[df_kinase['pdb_id'] == pdb_id] ## data
        ligand_name = list(data['ligand_name'])[0] ## data
        label = list(data['label'])[0]
        print(f'read file : {pdb_id}')

        try:
            ## [SAVE] SDF_wash.sdf
            original_data = f'{data_dir}/{pdb_id}/{pdb_id}_ligand_wash.sdf'
            savepath = f'{save_dir}/{pdb_id}/SDF_wash/SDF_wash.sdf'
            write_sdf(original_data, ligand_name, savepath)

            ## [SAVE] assay.csv
            savepath = f'{save_dir}/{pdb_id}/assay.csv'
            df_assay = pd.DataFrame([ligand_name, label]).T
            df_assay.to_csv(savepath, header=False, index=False, sep='\t')

            ## [SAVE] interpro.txt
            savepath = f'{save_dir}/{pdb_id}/interpro.txt'
            f = open(savepath, 'w')
            f.close()

            ## [SAVE] protein.fa
            savepath = f'{save_dir}/{pdb_id}/protein.fa'
            seq = pdb_to_fasta(data_dir, pdb_id)
            save_fasta(save_dir, pdb_id, seq)

        except:
            print(f'[ERROR] {pdb_id}')
            shutil.rmtree(f'{save_dir}/{pdb_id}/')
            er.append(pdb_id)
    print(f"# error list : {er}")
    return


#%%
def processing_reference(data_list, assay_index_name_path, assay_index_data_path, uniprot_kinase_list_path, label_th, savepath):
    def load_pl_name(assay_index_name_path):
        ## read file
        f = open(assay_index_name_path, 'r', encoding='UTF-8')
        list = [s for s in f.readlines() if not s.startswith('#')]
        pdb_id = [l[:4] for l in list]
        #release_year = [l[4:10].strip() for l in list]
        uniprot_id = [l[12:20].strip() for l in list]
        protein_name = [l[21:].strip() for l in list]
        f.close()
        pl_name = pd.DataFrame([pdb_id, uniprot_id, protein_name], index=['pdb_id', 'Uniprot ID', 'protein name']).T
        return pl_name

    def load_pl_data(assay_index_data_path):
        ## read file ===========-
        f = open(assay_index_data_path, 'r', encoding='UTF-8')
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

    def get_label(data_, th=30):
        ## data cleaning
        a = data_[data_["af_rel"] == "="]
        b = data_[(data_['af_rel'] == '<') & (data_['af_val'] <= th)]  # n=149 30μM以上で削除
        c = data_[(data_['af_rel'] == '<=') & (data_['af_val'] <= th)] # n=7
        d = data_[data_["af_rel"] == "~"]
        e = data_[(data_['af_rel'] == '>') & (data_['af_val'] > th)] # n=215 30μM以下で削除
        f = data_[(data_['af_rel'] == '>=') & (data_['af_val'] > th)] # n=1 削除
        ## merge
        data = pd.concat([a,b,c,d,e,f])

        ## label
        label = []
        for val in data['af_val'].tolist():
            if val < th:
                label.append(1) # active
            elif val >= th:
                label.append(0) # inactive

        ## add
        data['label'] = label

        return data

    def extract_kinase(df, uniprot_kinase_list_path):
        ## extract only kinase
        uniprot_name_kinase = pd.read_csv(uniprot_kinase_list_path, sep='\t')
        df_kinase = pd.merge(df, uniprot_name_kinase, left_on="Uniprot ID", right_on="Entry", how="left").dropna(how='any')
        return df_kinase

    ## load assay result data
    df_name = load_pl_name(assay_index_name_path)
    df_data_ = load_pl_data(assay_index_data_path)
    df_data = get_label(df_data_, th=label_th) ## label
    df_ = pd.merge(df_data, df_name, on='pdb_id', how='left')
    df = df_.query(f'pdb_id == {data_list}')
    df_kinase =  extract_kinase(df, uniprot_kinase_list_path).reset_index(drop=True)
    print(f'[INFO] data count={len(df_kinase)}, active={len(df_kinase[df_kinase["label"]==1])}, inactive={len(df_kinase[df_kinase["label"]==0])}')
    uniprot_name_list = len(df_kinase["Entry Name"].unique())
    print(f'# {uniprot_name_list} unique protein')
    df_kinase.to_csv(savepath)
    print(f"[SAVE] {savepath}")
    return df_kinase


#%%
def write_sdf(original_data, ligand_name, savepath):
    ## read sdf
    mols = [mol for mol in Chem.SDMolSupplier(original_data) if mol is not None]
    ## set prop
    mols[0].SetProp("_Name",ligand_name)
    ## write sdf
    writer=Chem.SDWriter(savepath)
    for mol in mols:
        writer.write(mol)
    writer.close()
    return


#%%
def pdb_to_fasta(data_dir, seq_name):
    # datapath
    datapath = f'{data_dir}/{seq_name}/{seq_name}_protein.pdb'

    if glob.glob(datapath) is not None:
        ## set amino code
        amino_code = {
                    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D',
                    'CYS':'C', 'GLN':'Q', 'GLU':'E', 'GLY':'G',
                    'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M',
                    'PHE':'F', 'PRO':'P', 'SER':'S', 'THR':'T',
                    'TRP':'W', 'TYR':'Y', 'VAL':'V', 'HIS':'H',
                    'ASX':'B', 'GLX':'Z', 'UNK':'K'}

        # structure > model > chain > residue > atom の構造
        p = PDBParser(QUIET = True)
        structure = p.get_structure(seq_name, datapath)

        seq = ''
        for model in structure:
            for chain in model:
                for residue in chain:
                    aaa = residue.get_resname()
                    if (aaa in amino_code):
                        seq += amino_code[aaa]

    return seq

#%%
def save_fasta(save_dir, seq_name, seq):
    savepath = f'{save_dir}/{seq_name}/protein.fa'

    seq_r = SeqRecord(Seq(seq)) # seq_recordに一旦格納
    seq_r.id = seq_name
    SeqIO.write(seq_r, savepath, "fasta")

    #print(f"[SAVE] fasta file {seq_name}")
    return


#%%
if __name__ == '__main__':
    main()

#%%
'''
data_dir = "/data_st02/drug/inoue/CGBVS/data/other/PDBbind/refined-set"
save_dir = "/data_st02/drug/inoue/CGBVS/data/original/pdbbind-refined"
label_th = 30
assay_index_name_path = "/data_st02/drug/inoue/CGBVS/data/other/PDBbind/index/INDEX_refined_name.2020"
assay_index_data_path = "/data_st02/drug/inoue/CGBVS/data/other/PDBbind/index/INDEX_refined_data.2020"
uniprot_kinase_list_path = '/data_st02/drug/inoue/CGBVS/data/other/uniprot/uniprot-query_kinase-protein-superfamily.tsv'


'''

# %%
