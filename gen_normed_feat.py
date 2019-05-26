from pathlib import Path
import numpy as np
from tqdm import tqdm

path = Path("/userhome/bigdata/train/visit_feat")
outpath = Path("/userhome/bigdata/train/visit_feat_normed")

if not outpath.exists():
    outpath.mkdir()

visit_files = path.iterdir()
mean = np.load("/userhome/bigdata/train/visit_mean.npy")
std = np.load("/userhome/bigdata/train/visit_std.npy")

std[np.where(std == 0)] = 1
# np.set_printoptions(threshold=99999999999)
for i, visit_file in tqdm(enumerate(visit_files)):
    npy = np.load(visit_file)
    # print(npy)
    npy = (npy - mean) / std
    # print(npy)
    # print(outpath / visit_file.name)
    np.save(outpath / visit_file.name, npy)

print("done")
