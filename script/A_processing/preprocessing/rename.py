
#%%
import os
import glob

#%%
def main():
    # config
    dataset_name = 'pdbbind-sample'
    #seq_name = '1a1e'

    data_dir = f'/data_st02/drug/inoue/CGBVS/data/original/{dataset_name}'



    seq_list = [i.split("/")[-1] for i in glob.glob(f"{data_dir}/*")]

    for seq_name in seq_list:

        os.makedirs(f'{data_dir}/{seq_name}/SDF_wash', exist_ok=True)

        path1 = f'{data_dir}/{seq_name}/{seq_name}_ligand_wash.sdf' # 変更前ファイル
        path2= f'{data_dir}/{seq_name}/SDF_wash/SDF_wash.sdf' # 変更後ファイル

        if os.path.exists(path1):
            # ファイル名の変更
            os.rename(path1, path2)
            print(f"[SAVE] rename {seq_name}")

        elif os.path.exists(path2):
            print("# already file exist")

        else:
            print("[ERROR] cannot find SDF file")

    return



# %%
if __name__ == '__main__':
    main()
# %%
