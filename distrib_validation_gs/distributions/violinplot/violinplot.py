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
 #input("i_run: ")
#print("Loading dataset", i_run, "...")
#firing_rate = get_all(pd.read_csv("firing_rate/run"+str(i_run)+".csv"), "fr", 0.0, 100.0)
#cv_isi = get_all(pd.read_csv("cv_isi/run"+str(i_run)+".csv"), "cv_isi", 0.0, 5.0)
#correlation = get_all(pd.read_csv("correlation/run"+str(i_run)+".csv"), "corr", -0.05, 0.2)

#merged datasets (requires more RAM)
#print("Loading merged dataset...")


print("Plotting...")
cifre=25
titolo=25
layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
#colore="Set2"
colors = ['#1A85FF','#DC3220']
sns.set_palette(sns.color_palette(colors))

#plt.suptitle("metastable state", fontsize = titolo + 3)

def plot_dataset(dataset, string, distid):
    cifre=14
    titolo=15
    #text coordinates
    left, width = 0.004, .5 
    bottom, height = .0, .83
    right = left + width
    top = bottom + height
    #colore="Set2"
    fig, axes = plt.subplots(8, 4)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.99, wspace=0.3, hspace=0.35) 
    layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                'STPa', '46', 'AITd', 'TH']
    plt.sca(axes[7, 3])
    #fig.suptitle(r'EMD '+string, fontsize=titolo+5)
    sns.set_theme(style="whitegrid")
    colors = ['#fc6333','#33BBEE']
    sns.set_palette(sns.color_palette(colors))
    if(string == "correlation"):
        gridsize = 400
    else:
        gridsize = 300
    for j in range(8):
        for i in range(4):
            area = (j*4)+i
            print("Plotting area {}".format(area))
            if(area!=31):
                popid_min = area*8
                popid_max = area*8 + 7
                data = dataset[dataset["popid"]>=popid_min]
                data = dataset[dataset["popid"]<=popid_max]
                sns.violinplot(ax=axes[j,i], x="layerid", y=distid, hue="Simulator", data=data, split=True, inner="quartile", cut=0.0, gridsize=gridsize, bw="silverman")
                axes[j, i].text(left, top, "Area "+str(1+i+j*4)+" ("+area_list[i+j*4]+")", 
                                fontsize= cifre+2, transform=axes[j, i].transAxes,
                                fontweight = "semibold", alpha = 0.6)
                axes[j, i].set_xticks(np.arange(len(layer)))
                axes[j, i].set_xticklabels(layer)
                axes[j, i].get_legend().remove()
                axes[j, i].set_xlabel("")
                if(string == "firing rate"):
                    axes[j, i].set_ylabel("rate [spikes/s]", fontsize=cifre+2)
                    axes[j, i].set_ylim([0,20])
                elif(string == "CV ISI"):
                    axes[j, i].set_ylabel("CV ISI", fontsize=cifre+2)
                    axes[j, i].set_ylim([0,2.0])
                else:
                    axes[j, i].set_ylabel("correlation", fontsize=cifre+2)
                    axes[j, i].set_ylim([-0.05, 0.2])
                
            else:
                popid_min = area*8
                data = dataset[dataset["popid"]>=popid_min]
                sns.violinplot(ax=axes[j,i], x="layerid", y=distid, hue="Simulator", data=data, split=True, inner="quartile", cut=0.0, gridsize=gridsize, bw="silverman")
                axes[j, i].text(left, top, "Area "+str(1+i+j*4)+" ("+area_list[i+j*4]+")", 
                                fontsize= cifre+2, transform=axes[j, i].transAxes,
                                fontweight = "semibold", alpha = 0.6)
                axes[j, i].set_xticks(np.arange(len(layer)-2))
                axes[j, i].set_xticklabels(['L2/3E', 'L2/3I', 'L5E', 'L5I', 'L6E', 'L6I'])
                plt.legend(bbox_to_anchor=(-1.5, -0.5), loc='lower center', borderaxespad=0.,
                            prop={'size': titolo+3}, title = "EMD "+string, title_fontsize = titolo +6, ncol=2)
                axes[j, i].set_xlabel("")
                if(string == "firing rate"):
                    axes[j, i].set_ylabel("rate [spikes/s]", fontsize=cifre+2)
                    axes[j, i].set_ylim([0,20])
                elif(string == "CV ISI"):
                    axes[j, i].set_ylabel("CV ISI", fontsize=cifre+2)
                    axes[j, i].set_ylim([0,2.0])
                else:
                    axes[j, i].set_ylabel("correlation", fontsize=cifre+2)
                    axes[j, i].set_ylim([-0.05, 0.15])
            axes[j, i].tick_params(labelsize=cifre)
            axes[j, i].grid(ls='--', axis='y')

    fig.set_size_inches(23.4, 33.1)
    plt.savefig("dist_violinplot_vert_ms_"+string.replace(" ", "_")+".pdf")
    plt.draw()
    return()


i_run = 8

#firing_rate = pd.read_csv("emd_firing_rate.csv")
data = pd.read_csv("firing_rate/run"+str(i_run)+".csv")
plot_dataset(data, "firing rate", "fr")


#cv_isi = pd.read_csv("emd_cv_isi.csv")
#data = pd.read_csv("cv_isi/run"+str(i_run)+".csv")
#plot_dataset(data, "CV ISI", "cv_isi")


#correlation = pd.read_csv("emd_corr.csv")
#data = pd.read_csv("correlation/run"+str(i_run)+".csv")
#plot_dataset(data, "correlation", "corr")




