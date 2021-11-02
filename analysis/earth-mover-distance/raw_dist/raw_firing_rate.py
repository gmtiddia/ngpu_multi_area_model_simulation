import numpy as np
import os.path
import elephant
import matplotlib.pyplot as plt
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef, covariance
from neo.core import SpikeTrain
from quantities import s
from eval_functions import __load_spike_times, __plot_hist, __smooth_hist
from scipy.stats import wasserstein_distance

nrun = 10
name = 'spike_times_'
begin = 1000.0
end = 11000.0
npop = 254

for i_run in range(nrun):
    print('Processing dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    path = 'path_to_data/data' + str(i_run) + '/spikes_pop_idx/'
    dum = []
    for i in range(npop):
        if(os.path.isfile("path_to_raw_dist/run"+str(i_run)+"/firing_rate/firing_rate_"+str(i)+".dat") == False):
            dum.append(i)

    if(len(dum)==0):
        print("The dataset " + str(i_run) + "is complete!", flush=True)
        continue
    else:
        print("Calculating distributions for population:", dum, flush=True)
        spike_times_list = __load_spike_times(path, name, begin, end, npop)

        for ipop in dum:
            print("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
            spike_times = spike_times_list[ipop]
            fr = []
            for st_row in spike_times:
                if len(st_row)==0:
                    fr.append(0.0)
                else:
                    fr.append(elephant.statistics.mean_firing_rate(np.array(st_row),
                                                                   begin/1000.0,
                                                                   end/1000.0))
                    
            file_data = open("path_to_dataset/run"+str(i_run)+"/firing_rate/firing_rate"+str(ipop)+".dat", "w")
            np.savetxt(file_data, fr)
            file_data.close()
