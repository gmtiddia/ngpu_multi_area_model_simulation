import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import os

y1mean = np.zeros((254, 300))
y2mean = np.zeros((254, 300))                 

path='../../data/'

for i_run in range(10):
    for i in range(254):
        fn1 = path+'mam_nest3_ms_validation/data' + str(i_run) + '/cv_isi_' + str(i) \
              + '.dat'
        fn2 = path+'mam_nestgpu_ms_validation/data' + str(i_run) + '/cv_isi_' + \
              str(i) + '.dat'


        
        data1 = np.loadtxt(fn1)
        data2 = np.loadtxt(fn2)

        x1=[row[0] for row in data1]
        y1=[row[1] for row in data1]

        x2=[row[0] for row in data2]
        y2=[row[1] for row in data2]

        if(len(y1)>0 and len(y2)>0):

            f1 = interpolate.interp1d(x1, y1,fill_value="extrapolate")
            f2 = interpolate.interp1d(x2, y2,fill_value="extrapolate")

            xnew = np.linspace(0, 5.0, 300)

            y1new = f1(xnew)
            y2new = f2(xnew)

            eps = 1.0e-3
            for j in range(len(y2new)):
                y2new[j] = max(y2new[j], eps)
                y1new[j] = max(y1new[j], eps)
                y1mean[i][j] = y1mean[i][j] + y1new[j]/10 
                y2mean[i][j] = y2mean[i][j] + y2new[j]/10

ticks=20
legenda=24
label=22
titolo=30
poplabel=["L2/3E", "L2/3I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
popname=["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                'STPa', '46', 'AITd', 'TH']
colors = ["#72b6a1", "#ea9474"]

for j in range(32):
    if j<31:
        fig=plt.figure(j+1)
        plt.suptitle("Area "+str(j+1)+" ("+area_list[j]+") CV ISI distribution", fontsize=titolo)
        for i in range(8):
            k=i+j*8
            plt.subplot(4,2,i+1).plot([0], [0], marker=' ', linestyle=' ', label=poplabel[i])
            plt.subplot(4,2,i+1).plot(xnew, y1mean[k], color=colors[0], label="NEST")
            plt.subplot(4,2,i+1).plot(xnew, y2mean[k], '--', color=colors[1], label="NEST GPU")
            plt.ylabel("p", fontsize=label)
            if i+1>6:
                plt.xlabel("CV ISI", fontsize=label)
            plt.tick_params(labelsize=ticks)
            plt.legend(fontsize=legenda)
            plt.grid()
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18)
        #plt.show()
        plt.savefig("../Areas/cv_isi/"+str(j+1), format='pdf')
        #input("Press Enter to continue...")
        #plt.close()
    else:
        fig=plt.figure(j+1)
        plt.suptitle("Area "+str(j+1)+" ("+area_list[j]+") CV ISI distribution", fontsize=titolo)
        poplabel=["L2/3E", "L2/3I", "L5E", "L5I", "L6E", "L6I"]
        popname=["L23E", "L23I", "L5E", "L5I", "L6E", "L6I"]
        for i in range(6):
            k=i+j*8
            plt.subplot(3,2,i+1).plot([0], [0], marker=' ', linestyle=' ', label=poplabel[i])
            plt.subplot(3,2,i+1).plot(xnew, y1mean[k], color=colors[0], label="NEST")
            plt.subplot(3,2,i+1).plot(xnew, y2mean[k], '--', color=colors[1], label="NEST GPU")
            plt.ylabel("p", fontsize=label)
            if i+1>4:
                plt.xlabel("CV ISI", fontsize=label)
            plt.tick_params(labelsize=ticks)
            plt.legend(fontsize=legenda)
            plt.grid()
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18)
        #plt.show()
        plt.savefig("../Areas/cv_isi/"+str(j+1), format='pdf')
        #input("Press Enter to continue...")
        #plt.close()
