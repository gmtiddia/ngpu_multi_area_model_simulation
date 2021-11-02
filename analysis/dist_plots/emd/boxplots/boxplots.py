import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def get_dataset(emd_array_root_name):
    emd_nest = np.loadtxt(emd_array_root_name+"_nest_nest.dat")
    emd_ngpu = np.loadtxt(emd_array_root_name+"_nest_ngpu.dat")
    npop=254
    nrun=10
    emd_list = []
    popid = []
    sim = []
    area = []
    for ipop in range(npop):
        dum_nest0 = emd_nest[ipop,:]
        dum_nest = []
        dum_ngpu0 = emd_ngpu[ipop,:]
        dum_ngpu = []
        for i in range(nrun):
            if(dum_nest0[i] != np.nan):
                dum_nest.append(dum_nest0[i])
                emd_list.append(dum_nest0[i])
            if(dum_ngpu0[i] != np.nan):
                dum_ngpu.append(dum_ngpu0[i])
                emd_list.append(dum_ngpu0[i])
        for i in range(len(dum_nest)):
            popid.append(ipop)
            sim.append("NEST-NEST")
            area.append(int(ipop/8))
        for i in range(len(dum_ngpu)):
            popid.append(ipop)
            sim.append("NEST-NeuronGPU")
            area.append(int(ipop/8))

    dataset = {"EMD": emd_list, "popid": popid, "Area": area, "Simulator": sim}
    data = pd.DataFrame(dataset)
    data.to_csv(emd_array_root_name+".csv", index=False)
    return(data)


def plot_dataset(dataset, string):
    cifre=14
    titolo=15
    #text coordinates
    left, width = 0.004, .5 
    bottom, height = .0, .83
    right = left + width
    top = bottom + height
    colore="Set2"
    fig, axes = plt.subplots(8, 4)
    plt.subplots_adjust(left=0.04, bottom=0.025, right=0.9, top=0.99, wspace=0.3, hspace=0.35) 
    layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                'STPa', '46', 'AITd', 'TH']
    plt.sca(axes[7, 3])
    #fig.suptitle(r'EMD '+string, fontsize=titolo+5)
    sns.set_theme(style="whitegrid")
    for j in range(8):
        for i in range(4):
            data = dataset[dataset["Area"]==(j*4)+i]
            if((i+j*4)!=31):
                sns.boxplot(ax=axes[j,i], x="popid", y="EMD", hue="Simulator", data=data, palette=colore)
                axes[j, i].text(left, top, "Area "+str(1+i+j*4)+" ("+area_list[i+j*4]+")", 
                                fontsize= cifre+2, transform=axes[j, i].transAxes,
                                fontweight = "semibold", alpha = 0.6)
                axes[j, i].set_xticks(np.arange(len(layer)))
                axes[j, i].set_xticklabels(layer)
                axes[j, i].get_legend().remove()
                axes[j, i].set_xlabel("")
                if(string == "firing rate"):
                    axes[j, i].set_ylabel("EMD [Hz]", fontsize=cifre+2)
                else:
                    axes[j, i].set_ylabel("EMD", fontsize=cifre+2)
                
            else:
                sns.boxplot(ax=axes[j,i], x="popid", y="EMD", hue="Simulator", data=data, palette=colore)
                axes[j, i].text(left, top, "Area "+str(1+i+j*4)+" ("+area_list[i+j*4]+")", 
                                fontsize= cifre+2, transform=axes[j, i].transAxes,
                                fontweight = "semibold", alpha = 0.6)
                axes[j, i].set_xticks(np.arange(len(layer)-2))
                axes[j, i].set_xticklabels(['L2/3E', 'L2/3I', 'L5E', 'L5I', 'L6E', 'L6I'])
                plt.legend(bbox_to_anchor=(1.56, 5.), loc='center right', borderaxespad=0.,
                            prop={'size': titolo+3}, title = "EMD "+string, title_fontsize = titolo +6)
                axes[j, i].set_xlabel("")
                if(string == "firing rate"):
                    axes[j, i].set_ylabel("EMD [Hz]", fontsize=cifre+2)
                else:
                    axes[j, i].set_ylabel("EMD", fontsize=cifre+2)
            axes[j, i].tick_params(labelsize=cifre)
            axes[j, i].grid(ls='--', axis='y')

    fig.set_size_inches(32, 18)
    plt.savefig("emd_boxplot_"+string.replace(" ", "_")+".png")
    plt.draw()
    return()

def sample_plot(firing_rate, cv_isi, correlation, areaid):
    cifre=20
    titolo=23
    colore="Set2"
    left, width = 0.004, .5 
    bottom, height = .0, .83
    right = left + width
    top = bottom + height
    layer=['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']
    area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                'STPa', '46', 'AITd', 'TH']

    firing_rate = firing_rate[firing_rate["Area"]==areaid]
    cv_isi = cv_isi[cv_isi["Area"]==areaid]
    correlation = correlation[correlation["Area"]==areaid]

    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)

    ax1 = plt.subplot(gs[0, :2],)
    plt.title(r'EMD firing rate', fontsize=titolo+3)
    sns.boxplot(x="popid", y="EMD", hue="Simulator", data=firing_rate, palette=colore)
    ax1.text(left, top, "Area "+str(areaid+1)+" ("+area_list[areaid]+")", 
                                    fontsize= cifre+2, transform=ax1.transAxes,
                                    fontweight = "semibold", alpha = 0.6)
    ax1.set_xlabel("")
    ax1.set_ylabel("EMD [Hz]", fontsize=cifre+2)
    ax1.set_xticks(np.arange(len(layer)))
    ax1.set_xticklabels(layer)
    ax1.tick_params(labelsize=cifre)
    plt.grid(ls='--', axis='y')
    plt.legend([],[], frameon=False)

    ax2 = plt.subplot(gs[0, 2:])
    plt.title(r'EMD CV ISI', fontsize=titolo+3)
    sns.boxplot(x="popid", y="EMD", hue="Simulator", data=cv_isi, palette=colore)
    ax2.text(left, top, "Area "+str(areaid+1)+" ("+area_list[areaid]+")", 
                                    fontsize= cifre+2, transform=ax2.transAxes,
                                    fontweight = "semibold", alpha = 0.6)
    ax2.set_xlabel("")
    ax2.set_ylabel("EMD", fontsize=cifre+2)
    ax2.set_xticks(np.arange(len(layer)))
    ax2.set_xticklabels(layer)
    plt.grid(ls='--', axis='y')
    ax2.tick_params(labelsize=cifre)
    plt.legend([],[], frameon=False)

    ax3 = plt.subplot(gs[1, 1:3])
    plt.title(r'EMD correlation', fontsize=titolo+3)
    sns.boxplot(x="popid", y="EMD", hue="Simulator", data=correlation, palette=colore)
    ax3.text(left, top, "Area "+str(areaid+1)+" ("+area_list[areaid]+")", 
                                    fontsize= cifre+2, transform=ax3.transAxes,
                                    fontweight = "semibold", alpha = 0.6)
    ax3.set_xlabel("")
    ax3.set_ylabel("EMD", fontsize=cifre+2)
    ax3.set_xticks(np.arange(len(layer)))
    ax3.set_xticklabels(layer)
    plt.grid(ls='--', axis='y')
    ax3.tick_params(labelsize=cifre)
    plt.legend(bbox_to_anchor=(1.5, 0.5),loc='center right', prop={'size': titolo})
    plt.draw()



#firing_rate = get_dataset("emd_firing_rate")
firing_rate = pd.read_csv("emd_firing_rate.csv")
#plot_dataset(firing_rate, "firing rate")

#cv_isi = get_dataset("emd_cv_isi")
cv_isi = pd.read_csv("emd_cv_isi.csv")
#plot_dataset(cv_isi, "CV ISI")

correlation = pd.read_csv("emd_corr.csv")
#correlation = get_dataset("emd_corr")
#plot_dataset(correlation, "correlation")


sample_plot(firing_rate, cv_isi, correlation, 0)
plt.show()






