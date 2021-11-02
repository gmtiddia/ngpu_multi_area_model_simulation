import numpy as np
import elephant
import matplotlib.pyplot as plt
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef, covariance
from neo.core import SpikeTrain
from quantities import s
from scipy.stats import wasserstein_distance
import pandas as pd
import os

nrun = 10
npop = 254

for i_run in range(nrun):
    print ('Loading dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    path1 = 'path_to_dataset1/run' + str(i_run) + '/cv_isi/'
    path2 = 'path_to_dataset2/run' + str(i_run) + '/cv_isi/'
    cvisilist = []
    popid = []
    sim = []
    for ipop in range(npop):
        if(os.path.isfile(path1+"cv_isi_"+str(ipop)+".dat")==True and os.path.isfile(path2+"cv_isi_"+str(ipop)+".dat")==True):
            print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
            cv_isi1 = np.loadtxt(path1+"cv_isi_"+str(ipop)+".dat")
            cv_isi2 = np.loadtxt(path2+"cv_isi_"+str(ipop)+".dat")
            if(cv_isi1.size>1 and cv_isi2.size>1):
                cvisilist += [i for i in cv_isi1]
                cvisilist += [i for i in cv_isi2]
                sim += ["NEST" for i in range(len(cv_isi1))]
                sim += ["NeuronGPU" for i in range(len(cv_isi2))]
                if(ipop<250):
                    popid += [ipop%8 for i in range(len(cv_isi1))]
                    popid += [ipop%8 for i in range(len(cv_isi2))]
                else:
                    ipop=ipop+2
                    popid += [ipop%8 for i in range(len(cv_isi1))]
                    popid += [ipop%8 for i in range(len(cv_isi2))]
    dataset = {"cv_isi": cvisilist, "popid": popid, "Simulator": sim}
    data = pd.DataFrame(dataset)
    data.to_csv("path/run"+str(i_run)+".csv", index=False)
    