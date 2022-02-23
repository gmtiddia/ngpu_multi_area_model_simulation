import numpy as np
import os
import elephant
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr


def __gather_metadata(path):
    """ Reads first and last ids of
    neurons in each population.

    Parameters
    ------------
    path
        Path where the spike detector files are stored.

    Returns
    -------
    node_ids
        Lowest and highest id of nodes in each population.

    """
    # load node IDs
    node_idfile = open(path + 'population_nodeids.dat', 'r')
    node_ids = []
    for l in node_idfile:
        node_ids.append(l.split())
    node_ids = np.array(node_ids, dtype='i4')
    return node_ids


def __load_spike_times(path, name, begin, end, npop):
    """ Loads spike times of each spike detector.

    Parameters
    ----------
    path
        Path where the files with the spike times are stored.
    name
        Name of the spike detector.
    begin
        Time point (in ms) to start loading spike times (included).
    end
        Time point (in ms) to stop loading spike times (included).

    Returns
    -------
    data
        Dictionary containing spike times in the interval from ``begin``
        to ``end``.

    """
    node_ids = __gather_metadata(path)
    data = {}
    dtype = {'names': ('sender', 'time_ms'),  # as in header
             'formats': ('i4', 'f8')}

    sd_names = {}
    
    for i_pop in range(npop):
        fn = os.path.join(path, 'spike_times_' + str(i_pop) + '.dat')
        data_i_raw = np.loadtxt(fn, skiprows=1, dtype=dtype)
        #data_i_raw = np.loadtxt(fn, dtype=dtype)
        #data_i_raw = np.sort(data_i_raw, order='time_ms')
        # begin and end are included if they exist
        #low = np.searchsorted(data_i_raw['time_ms'], v=begin, side='left')
        #high = np.searchsorted(data_i_raw['time_ms'], v=end, side='right')
        data[i_pop] = data_i_raw #[low:high]
        sd_names[i_pop] = 'spike_times_' + str(i_pop)

    spike_times_list = []
    for i_pop, name_pop in enumerate(sd_names):
        spike_times = []
        for id in np.arange(node_ids[i_pop, 0], node_ids[i_pop, 1] + 1):
            spike_times.append([])
        if data[i_pop].size>1:
            #data[i_pop].size>0
            for row in data[i_pop]:
                time_ms = row[1]
                if time_ms>=begin and time_ms<end:
                    sender = row[0]
                    time = time_ms/1000.0
                    i_neur = sender - node_ids[i_pop, 0]
                    spike_times[i_neur].append(time)
            
        spike_times_list.append(spike_times)

    return spike_times_list


# plot histogram
def __plot_hist(data, xmin, xmax, ax):
    hist, bins = np.histogram(data, bins='fd', range=(xmin, xmax))
    x = []
    for i in range(len(bins)-1):
        x.append((bins[i+1]+bins[i])/2.0)
        
    ax.plot(x, hist)
    ax.set_xlim([xmin, xmax])

# smooth histogram using kernal density
def __smooth_hist(data, xmin, xmax, nx, bw_min=None):
    x = np.linspace(xmin, xmax, nx)

    # convert data to list of 1-element lists
    y=[]
    for elem in data:
        y.append([elem])

    # Silverman rule for kernel density bandwidth
    bw = 0.9*min(np.std(data), iqr(data)/1.34) \
         *(1.0*len(data))**(-0.2)

    if bw_min != None:
        bw = max(bw, bw_min)

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(y)

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(np.array(x)[:, None])
    hist_smooth = np.exp(logprob)

    # Normalize histogram
    hist = hist_smooth / np.sum(hist_smooth) \
           / (x[1] - x[0])
    return x, hist

def __show_all():
    plt.draw()
    plt.pause(0.5)
    input("Press Enter to continue...")
    plt.close()
