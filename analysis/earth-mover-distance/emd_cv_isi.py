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
    path1 = 'path_to_dataset1/run' + str(i_run) + '/cv_isi/'
    path2 = 'path_to_dataset2/run' + str(i_run) + '/cv_isi/'
    for ipop in range(npop):
        print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
        cv_isi1 = np.loadtxt(path1+"cv_isi_"+str(ipop)+".dat")
        cv_isi2 = np.loadtxt(path2+"cv_isi_"+str(ipop)+".dat")
        
        if(cv_isi1.size > 1 and cv_isi2.size > 1):
            emd[ipop, i_run] = wasserstein_distance(cv_isi1,cv_isi2)
        else:
            emd[ipop, i_run] = np.nan
            
np.savetxt("emd_cv_isi.dat", emd)
