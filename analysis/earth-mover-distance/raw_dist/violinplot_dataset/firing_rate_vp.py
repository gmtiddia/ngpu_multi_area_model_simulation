import numpy as np
import elephant
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    path1 = 'path_to_dataset1/run' + str(i_run) + '/firing_rate/'
    path2 = 'path_to_dataset2/run' + str(i_run) + '/firing_rate/'
    frlist = []
    popid = []
    sim = []
    for ipop in range(npop):
        print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
        fr1 = np.loadtxt(path1+"firing_rate_"+str(ipop)+".dat")
        fr2 = np.loadtxt(path2+"firing_rate_"+str(ipop)+".dat")
        frlist += [i for i in fr1]
        frlist += [i for i in fr2]
        sim += ["NEST" for i in range(len(fr1))]
        sim += ["NEST GPU" for i in range(len(fr2))]
        if(ipop<250):
            popid += [ipop%8 for i in range(len(fr1))]
            popid += [ipop%8 for i in range(len(fr2))]
        else:
            ipop=ipop+2
            popid += [ipop%8 for i in range(len(fr1))]
            popid += [ipop%8 for i in range(len(fr2))]
    dataset = {"fr": frlist, "popid": popid, "Simulator": sim}
    data = pd.DataFrame(dataset)
    data.to_csv("path/run"+str(i_run)+".csv", index=False)
