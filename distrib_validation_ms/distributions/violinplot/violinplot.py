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
i_run = 0 #input("i_run: ")
#print("Loading dataset", i_run, "...")

#merged datasets (requires more RAM)
#print("Loading merged dataset...")



print("Plotting...")
cifre=25
titolo=25
layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
#colore="Set2"
colors = ['#1A85FF','#DC3220']
sns.set_palette(sns.color_palette(colors))

fig=plt.figure()
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)

plt.suptitle("metastable state", fontsize = titolo + 3)

print("Loading Firing Rate")
#firing_rate = get_all(pd.read_csv("firing_rate/firing_rate.csv"), "fr", 0.0, 100.0)
firing_rate = get_all(pd.read_csv("firing_rate/run"+str(i_run)+".csv"), "fr", 0.0, 100.0)


ax1 = plt.subplot(gs[0, :2],)
v1 = sns.violinplot(x="popid", y="fr", hue="Simulator", data=firing_rate, split=True, inner="quartile",
cut=0.0, gridsize=300, bw='silverman')
for l in v1.lines:
    #l.set_linestyle('--')
    l.set_linewidth(1.5)
    l.set_color('black')
    #l.set_alpha(0.8)
for l in v1.lines[1::3]:
    #l.set_linestyle('--')
    l.set_linewidth(1.5)
    l.set_color('black')
    #l.set_alpha(0.8)
plt.xticks(np.arange(len(layer)), layer)
plt.xlabel("")
plt.ylabel("Firing rate [spikes/s]", size=titolo)
plt.ylim(-0.5, 40)
plt.text(-0.125,1.03,"A", transform=ax1.transAxes, weight="bold", fontsize=titolo)
plt.tick_params(labelsize=cifre)
plt.grid()
plt.legend([],[], frameon=False)

del firing_rate
print("Loading CV ISI")
#cv_isi = get_all(pd.read_csv("cv_isi/cv_isi.csv"), "cv_isi", 0.0, 5.0)
cv_isi = get_all(pd.read_csv("cv_isi/run"+str(i_run)+".csv"), "cv_isi", 0.0, 5.0)

ax2 = plt.subplot(gs[0, 2:])
v2 = sns.violinplot(x="popid", y="cv_isi", hue="Simulator", data=cv_isi, split=True, inner="quartile",
cut=0.0, gridsize=300, bw="silverman")
for l in v2.lines:
    #l.set_linestyle('--')
    l.set_linewidth(1.5)
    l.set_color('black')
    #l.set_alpha(0.8)
for l in v2.lines[1::3]:
    #l.set_linestyle('--')
    l.set_linewidth(1.5)
    l.set_color('black')
    #l.set_alpha(0.8)
plt.xticks(np.arange(len(layer)), layer)
plt.xlabel("")
plt.ylabel("CV ISI", size=titolo)
plt.ylim(-0.05, 5.5)
plt.text(-0.125,1.03,"B", transform=ax2.transAxes, weight="bold", fontsize=titolo)
plt.tick_params(labelsize=cifre)
plt.grid()
plt.legend([],[], frameon=False)

del cv_isi
print("Loading Correlation")
#correlation = get_all(pd.read_csv("correlation/correlation.csv"), "corr", -0.05, 0.2)
correlation = get_all(pd.read_csv("correlation/run"+str(i_run)+".csv"), "corr", -0.05, 0.2)

ax3 = plt.subplot(gs[1, 1:3])
v3 = sns.violinplot(x="popid", y="corr", hue="Simulator", data=correlation, split=True, inner="quartile",
cut=0.0, gridsize=400, bw="silverman")
for l in v3.lines:
    #l.set_linestyle('--')
    l.set_linewidth(1.5)
    l.set_color('black')
    #l.set_alpha(0.8)
for l in v3.lines[1::3]:
    #l.set_linestyle('--')
    l.set_linewidth(1.5)
    l.set_color('black')
    #l.set_alpha(0.8)
plt.xticks(np.arange(len(layer)), layer)
plt.xlabel("")
plt.ylabel("correlation", size=titolo)
plt.ylim(-0.030, 0.15)
plt.text(-0.125,1.03,"C", transform=ax3.transAxes, weight="bold", fontsize=titolo)
plt.tick_params(labelsize=cifre)
plt.grid()
#plt.show()

del correlation
print("Plot")

fig.set_size_inches(32, 18)
plt.subplots_adjust(top=0.9, hspace = 0.25)
#plt.savefig("violinplot_prova.png")
plt.show()