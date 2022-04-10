import numpy as np
import elephant
import matplotlib.pyplot as plt
from elephant.statistics import isi, cv
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef, covariance
from neo.core import SpikeTrain
from quantities import s
from eval_functions import __load_spike_times, __plot_hist, __smooth_hist


nrun = 10
name = 'spike_times_'
begin = 1000.0
end = 11000.0
npop = 254
xmin = 0.0
xmax = 15.0
nx = 300

matrix_size = 200
spike_time_bin = 0.002

for i_run in range(nrun):
    print ('Processing dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    path = '../data' + str(i_run) + '/spikes_pop_idx/'
    spike_times_list = __load_spike_times(path, name, begin, end, npop)

    for ipop in range(npop):
        print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
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

       
#        if i_run==0 and ipop<8:
#            plt.figure(ipop)
#            fig, (ax1, ax2) = plt.subplots(2)
#            __plot_hist(fr, xmin, xmax, ax1)
#            
#            ax2.plot(x, hist1)
#            ax2.set_xlim([xmin, xmax])
#
#plt.draw()
#plt.pause(0.5)
#input("Press Enter to continue...")
#plt.close()
