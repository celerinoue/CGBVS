
#%%
#from lib2to3.pgen2.pgen import DFAState
import pandas as pd
import numpy as np
import argparse
import sys
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
from Bio.PDB import *


#%%
def main():
    ## config & args
    data_dir = sys.argv[0]
    save_dir = sys.argv[1]

    ## run
    seq_list = [i.split("/")[-1] for i in glob.glob(f"{data_dir}/*")]
    for seq_name in seq_list:
        ## make dir
        os.makedirs(f'{save_dir}/{seq_name}')
        ## run
        seq = pdb_to_fasta(data_dir, seq_name)
        save_fasta(save_dir, seq_name, seq)
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


def save_fasta(save_dir, seq_name, seq):
    savepath = f'{save_dir}/{seq_name}/protein.fa'

    seq_r = SeqRecord(Seq(seq)) # seq_recordに一旦格納
    seq_r.id = seq_name
    SeqIO.write(seq_r, savepath, "fasta")

    print(f"[SAVE] fasta file {seq_name}")
    return



# %%
if __name__ == '__main__':
    main()
# %%
