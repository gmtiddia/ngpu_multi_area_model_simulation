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
xmin = -0.05
xmax = 0.15
nx = 400

matrix_size = 200
spike_time_bin = 0.002

for i_run in range(nrun):
    print ('Processing dataset '+ str(i_run+1) + '/' + str(nrun), flush=True)
    path = '../data' + str(i_run) + '/spikes_pop_idx/'
    spike_times_list = __load_spike_times(path, name, begin, end, npop)

    for ipop in range(npop):
        print ("run ", i_run, "/", nrun, "  pop ", ipop, "/", npop, flush=True)
        spike_times = spike_times_list[ipop]
        st_list = []
        for j in range(matrix_size):
            spike_train = SpikeTrain(np.array(spike_times[j])*s,
                                     t_stop = (end/1000.0)*s)
            st_list.append(spike_train)
                                     
        binned_st = BinnedSpikeTrain(st_list, spike_time_bin*s, None,
                                     (begin/1000.0)*s, (end/1000.0)*s)
        #print (binned_st)
        cc_matrix = corrcoef(binned_st)
        correl = []
        for j in range(matrix_size):
            for k in range(matrix_size):
                #print(j, k, cc_matrix[j][k])
                if (j != k and cc_matrix[j][k]<xmax and cc_matrix[j][k]>xmin):
                    correl.append(cc_matrix[j][k])

        if len(correl)>0:
            #bw_min = (xmax - xmin)/1.0e4
            x, hist1 =  __smooth_hist(correl, xmin, xmax, nx) #, bw_min)
            arr = np.column_stack((x, hist1))
            np.savetxt(path+'correl_'+str(ipop)+'.dat', arr)
       
#        if i_run==0:
#            plt.figure(ipop)
#            fig, (ax1, ax2) = plt.subplots(2)
#            __plot_hist(correl, xmin, xmax, ax1)
#            
#            ax2.plot(x, hist1)
#            ax2.set_xlim([xmin, xmax])
#
#plt.draw()
#plt.pause(0.5)
#input("Press Enter to continue...")
#plt.close()
