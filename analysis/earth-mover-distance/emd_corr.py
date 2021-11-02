import numpy as np
import elephant
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef, covariance
from neo.core import SpikeTrain
from quantities import s
from scipy.stats import wasserstein_distance
import os.path

nrun = 10
npop = 254

emd=np.zeros((npop,nrun))

for i_run in range(nrun):
    print ('Loading dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    path1 = 'path_to_dataset1/run' + str(i_run) + '/correlation/'
    path2 = 'path_to_dataset2/run' + str(i_run) + '/correlation/'
    for ipop in range(npop):
        if(os.path.isfile(path1+"corr_"+str(ipop)+".dat")==True and os.path.isfile(path2+"corr_"+str(ipop)+".dat")==True):
            print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)

            correl1 = np.loadtxt(path1+"corr_"+str(ipop)+".dat")
            correl2 = np.loadtxt(path2+"corr_"+str(ipop)+".dat")
        
            if(len(correl1)>0 and len(correl2)>0):
                emd[ipop, i_run] = wasserstein_distance(correl1,correl2)
            else:
                emd[ipop,i_run] = np.nan
        else:
            emd[ipop,i_run] = np.nan
            
np.savetxt("emd_corr.dat", emd)
