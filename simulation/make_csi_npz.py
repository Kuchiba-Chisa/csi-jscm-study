"""把 COST2100 的 .mat 转成本项目的 npz 格式（data_csi/{train,val,test}.npz，键 x）。

HT: (N, 2048) float64，取值 [0,1]（已按 CsiNet 约定归一化，读取时减 0.5）。
2048 = 32*32*2，按 (32,32,2) 解释为 H,W,(re,im)，转到 (2,32,32)。
"""
import os

import numpy as np
import scipy.io as sio

MAT = "/root/work/csi2/mat"
OUT = "/root/work/jscm/data_csi"

os.makedirs(OUT, exist_ok=True)


def conv(path, n_take=None):
    ht = sio.loadmat(path)["HT"].astype(np.float32)          # (N, 2048)
    if n_take:
        ht = ht[:n_take]
    # CsiNet / RD-JSCC 的约定：每行前 1024 个是实部、后 1024 个是虚部（block，不是交错）。
    # RD-JSCC `data/datasets/cost2100.py:20` 用的就是 reshape(N, channel=2, nt=32, nc=32)。
    h = ht.reshape(len(ht), 2, 32, 32) - 0.5                 # N,2,32,32
    return np.ascontiguousarray(h)


tr = conv(f"{MAT}/DATA_Htrainin.mat", 12000)
te = conv(f"{MAT}/DATA_Htestin.mat", 2000)
va, te = te[:1000], te[1000:2000]

for name, arr in (("train", tr), ("val", va), ("test", te)):
    np.savez_compressed(f"{OUT}/{name}.npz", x=arr)
    print(name, arr.shape, "E|x|^2=%.4f" % float((arr.astype(np.float64) ** 2).mean()), flush=True)
print("saved to", OUT)
