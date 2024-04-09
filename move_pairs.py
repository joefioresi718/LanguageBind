import glob
import pandas as pd
import shutil
import os

import config as cfg


df = pd.read_csv('CLIPblind_pairs_k400_vitl_final.csv')

print(df.head())
vid_path = 'V-MMVP/k400_pairs'
os.makedirs(vid_path, exist_ok=True)

for i, row in df.iterrows():
    pair_path = f'{vid_path}/kin{i:03d}'
    os.makedirs(pair_path, exist_ok=True)
    v1 = row['video1']
    v2 = row['video2']
    # shutil.copy(f'{cfg.ucf101_path}/Videos/{v1.split("_")[1].replace("HandStand", "Handstand")}/{v1}', f'{pair_path}/{v1}')
    # shutil.copy(f'{cfg.ucf101_path}/Videos/{v2.split("_")[1].replace("HandStand", "Handstand")}/{v2}', f'{pair_path}/{v2}')
    shutil.copy(f'{cfg.kinetics_path}/test/{v1}', f'{pair_path}/{v1}')
    shutil.copy(f'{cfg.kinetics_path}/test/{v2}', f'{pair_path}/{v2}')

print('Done!')