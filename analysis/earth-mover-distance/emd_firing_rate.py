import numpy as np
import elephant
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef, covariance
from neo.core import SpikeTrain
from quantities import s
from scipy.stats import wasserstein_distance

nrun = 10
npop = 254

emd=np.zeros((npop,nrun))

for i_run in range(nrun):
    print ('Loading dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    path1 = 'path_to_dataset1/run' + str(i_run) + '/firing_rate/'
    path2 = 'path_to_dataset2/run' + str(i_run) + '/firing_rate/'
    for ipop in range(npop):
        print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
        fr1 = np.loadtxt(path1+"firing_rate_"+str(ipop)+".dat")
        fr2 = np.loadtxt(path2+"firing_rate_"+str(ipop)+".dat")
                
        emd[ipop, i_run] = wasserstein_distance(fr1,fr2)
              
np.savetxt("emd_firing_rate.dat", emd)  
