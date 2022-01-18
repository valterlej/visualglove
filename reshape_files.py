import glob
import numpy as np
from tqdm import tqdm

directory = "/home/vlejunior/datasets/i3d_acnet_kinetics/"

files = glob.glob(directory+"*.npy")

for f in tqdm(files):
    try:
        x = np.load(f)
        if x.shape[1] == 1152:
            x = x[:,0:1024]
            with open(f, 'wb') as npfile:
                np.save(npfile, x)
    except:
        print(f"file not loaded - {f}")