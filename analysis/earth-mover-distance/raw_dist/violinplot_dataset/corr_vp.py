import numpy as np
import os.path
import pandas as pd
import elephant
import matplotlib.pyplot as plt
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef, covariance
from neo.core import SpikeTrain
from quantities import s
from scipy.stats import wasserstein_distance

nrun = 10
npop = 254

for i_run in range(nrun):
    print ('Loading dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    path1 = 'path_to_dataset1/run' + str(i_run) + '/correlation/'
    path2 = 'path_to_dataset2/run' + str(i_run) + '/correlation/'
    corrlist = []
    popid = []
    sim = []
    for ipop in range(npop):
        if(os.path.isfile(path1+"corr_"+str(ipop)+".dat")==True and os.path.isfile(path2+"corr_"+str(ipop)+".dat")==True):
            print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
            correl1 = np.loadtxt(path1+"corr_"+str(ipop)+".dat")
            correl2 = np.loadtxt(path2+"corr_"+str(ipop)+".dat")
            if(len(correl1)>0 and len(correl2)>0):
                corrlist += [i for i in correl1]
                corrlist += [i for i in correl2]
                sim += ["NEST" for i in range(len(correl1))]
                sim += ["NEST GPU" for i in range(len(correl2))]
                if(ipop<250):
                    popid += [ipop%8 for i in range(len(correl1))]
                    popid += [ipop%8 for i in range(len(correl2))]
                else:
                    ipop=ipop+2
                    popid += [ipop%8 for i in range(len(correl1))]
                    popid += [ipop%8 for i in range(len(correl2))]
    dataset = {"corr": corrlist, "popid": popid, "Simulator": sim}
    data = pd.DataFrame(dataset)
    data.to_csv("path/run"+str(i_run)+".csv", index=False)
