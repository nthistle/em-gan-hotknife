import numpy as np
from matplotlib import pyplot as plt
import h5py
import json

data = h5py.File("/home/thistlethwaiten/data/hotknifedata.hdf5", "r")["/volumes/data"]
pts = []
for idx in range(data.shape[0]):
	pts.append(tuple(np.percentile(data[idx], [25,50,75])))
with open("hk-brightness.json","w") as f:
	json.dump(pts, f)
