from pathlib import Path
import numpy as np
from tqdm import tqdm
path = Path("/userhome/bigdata/train/visit_feat")

visit_files = path.iterdir()

c = np.zeros((40000, 7, 26, 24))
for i, visit_file in tqdm(enumerate(visit_files)):
    npy = np.load(visit_file)
    c[i] = npy

mean = np.mean(c, axis=0)
std = np.std(c, axis=0)

print(mean.shape)
print(std.shape)


np.save("/userhome/bigdata/train/visit_mean.npy", mean)
np.save("/userhome/bigdata/train/visit_std.npy", std)

print("done")