import brewer2mpl
import numpy as np
from matplotlib import pyplot as plt
import h5py
import json

bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

with open("hk-brightness.json") as f:
	pts1 = json.load(f)
with open("hk-brightness-2.json") as f:
	pts2 = json.load(f)
pts1 = np.array(pts1)
pts2 = np.array(pts2)

#params = {
#	'axes.labelsize': 8,
#	'text.fontsize': 8,
#	'legend.fontsize': 10,
#	'xtick.labelsize': 10,
#	'ytick.labelsize': 10,
#	'text.usetex': False,
#	}

#plt.rcParams.update(params)
x = np.linspace(0,len(pts1)-1,num=len(pts1))

#plt.ylim(0,255)
plt.ylim(100,250)

plt.axvline(x=101-32, color="#BBBBBB", linestyle='--')
plt.axvline(x=101+31, color="#BBBBBB", linestyle='--')

plt.axvline(x=101-16, color="#222222", linestyle='-')
plt.axvline(x=101+15, color="#222222", linestyle='-')

plt.fill_between(x, pts1[:,0], pts1[:,2], alpha=0.25, linewidth=0, color=colors[0])
plt.plot(x, pts1[:,1], color=colors[0])

plt.fill_between(x, pts2[:,0], pts2[:,2], alpha=0.25, linewidth=0, color=colors[1])
plt.plot(x, pts2[:,1], color=colors[1])

plt.show()
