import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import iqr
import os

def get_all(file, var_name, xmin, xmax):
    print(var_name + " dataset")
    file = file[file[var_name]>=xmin]
    file = file[file[var_name]<=xmax]
    return(file)

#one simulation data
i_run = 6 #{0..9}
print("Loading dataset", i_run, "...")
firing_rate = get_all(pd.read_csv("path_to_firing_rate_vp/run"+str(i_run)+".csv"), "fr", 0.0, 100.0)
cv_isi = get_all(pd.read_csv("path_to_cv_isi_vp/run"+str(i_run)+".csv"), "cv_isi", 0.0, 5.0)
correlation = get_all(pd.read_csv("path_to_correlation_vp/run"+str(i_run)+".csv"), "corr", -0.05, 0.2)

print("Plotting...")
cifre=20
titolo=23
layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
colore="Set2"

fig=plt.figure()
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)

ax1 = plt.subplot(gs[0, :2],)
sns.violinplot(x="popid", y="fr", hue="Simulator", data=firing_rate, split=True, inner="quartile",
palette=colore, cut=0.0, gridsize=300, bw='silverman')
plt.xticks(np.arange(len(layer)), layer)
plt.xlabel("")
plt.ylabel("Firing rate [Hz]", size=titolo)
plt.ylim(-0.5, 40)
plt.tick_params(labelsize=cifre)
plt.grid()
plt.legend([],[], frameon=False)

ax2 = plt.subplot(gs[0, 2:])
sns.violinplot(x="popid", y="cv_isi", hue="Simulator", data=cv_isi, split=True, inner="quartile",
palette=colore, cut=0.0, gridsize=300, bw="silverman")
plt.xticks(np.arange(len(layer)), layer)
plt.xlabel("")
plt.ylabel("CV ISI", size=titolo)
plt.ylim(-0.05, 5.5)
plt.tick_params(labelsize=cifre)
plt.grid()
plt.legend([],[], frameon=False)

ax3 = plt.subplot(gs[1, 1:3])
sns.violinplot(x="popid", y="corr", hue="Simulator", data=correlation, split=True, inner="quartile",
palette=colore, cut=0.0, gridsize=400, bw="silverman")
plt.xticks(np.arange(len(layer)), layer)
plt.xlabel("")
plt.ylabel("correlation", size=titolo)
plt.ylim(-0.030, 0.15)
plt.tick_params(labelsize=cifre)
plt.grid()
plt.legend(bbox_to_anchor=(1.5, 0.5),loc='center right', prop={'size': titolo})
#plt.show()
fig.set_size_inches(32, 18)
plt.savefig("violinplot.png")
