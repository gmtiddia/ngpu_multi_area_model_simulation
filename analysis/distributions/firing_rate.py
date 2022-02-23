import numpy as np
import elephant
import matplotlib.pyplot as plt
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef, covariance
from neo.core import SpikeTrain
from quantities import s
from eval_functions import __load_spike_times, __plot_hist, __smooth_hist
import os

nrun = 10
name = 'spike_times_'
begin = 500.0
end = 10500.0
npop = 254
xmin = 0.0
xmax = 100.0
nx = 300

matrix_size = 200
spike_time_bin = 0.002

for i_run in range(nrun):
    print('Processing dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    path = 'path_to_data/data' + str(i_run) + '/spikes_pop_idx/'
    dum = []
    for i in range(npop):
        if(os.path.isfile(path+'firing_rate_'+str(i)+'.dat') == False):
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
        
            bw_min = 0.5*1000.0/(end - begin)
            x, hist1 =  __smooth_hist(fr, xmin, xmax, nx, bw_min)
            arr = np.column_stack((x, hist1))
            np.savetxt(path+'firing_rate_'+str(ipop)+'.dat', arr)
