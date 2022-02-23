import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datapath = ""
name_times = ['Building', 'Simulation', 'neuron_Update_time',
'poisson_generator_time', 'GetSpike_time', 'NestedLoop_time', 'SpikeBufferUpdate_time',
'SpikeReset_time', 'copy_ext_spike_time', 'SendExternalSpike_time', 'SendSpikeToRemote_time',
'RecvSpikeFromRemote_time', 'ExternalSpikeReset_time']

label = []
times = []

for filename in name_times:
    fn = datapath+"/all_"+filename+".dat"
    file = np.loadtxt(fn)
    #print("File "+filename+": ", file[1::2])
    times.append(file[1::2])
    #compute names
    if (filename == "Building"):
        label.append("Building time")
    elif (filename == "Simulation"):
        label.append("Simulation time")
    else:
        label.append(filename)

data = {}
for i in range(len(label)):
    data[label[i]] = times[i]

df = pd.DataFrame(data)
df.to_csv("raw_times.csv")

bt = []
st = []
neuron = []
poisson = []
remote_spike = []
local_spike = []
other = []

for i in range(10):
    #take each run results
    run_i = df[i*32:(i+1)*32]
    #extract max building time and simulation time
    max_bt = max(run_i["Building time"])
    bt.append(max_bt)
    max_st = max(run_i["Simulation time"])
    st.append(max_st)
    #extract the indexed that corresponds to the highest simulation times
    max_st_idx = run_i.index[run_i["Simulation time"] == max_st].tolist()
    #sometimes there is more than one node that exibits the maximum value
    #in this case we average the results in order to show a single set of values per each node
    if(len(max_st_idx) == 1):
        remote_spike_handling_delivery = (run_i["copy_ext_spike_time"].loc[max_st_idx[0]] +
            run_i["SendExternalSpike_time"].loc[max_st_idx[0]] +
            run_i["SendSpikeToRemote_time"].loc[max_st_idx[0]] +
            run_i["RecvSpikeFromRemote_time"].loc[max_st_idx[0]] +
            run_i["ExternalSpikeReset_time"].loc[max_st_idx[0]])
        local_spike_handling_delivery = (run_i["GetSpike_time"].loc[max_st_idx[0]] +
            run_i["NestedLoop_time"].loc[max_st_idx[0]] +
            run_i["SpikeBufferUpdate_time"].loc[max_st_idx[0]] +
            run_i["SpikeReset_time"].loc[max_st_idx[0]])
        oth = (run_i["Simulation time"].loc[max_st_idx[0]] -
            np.sum(run_i.drop(columns=["Building time", "Simulation time"]).loc[max_st_idx[0]]))
        neuron.append(run_i["neuron_Update_time"].loc[max_st_idx[0]])
        poisson.append(run_i["poisson_generator_time"].loc[max_st_idx[0]])
        remote_spike.append(remote_spike_handling_delivery)
        local_spike.append(local_spike_handling_delivery)
        other.append(oth)
    else:
        remote_spike_handling_delivery = (np.mean(run_i["copy_ext_spike_time"].loc[max_st_idx]) +
            np.mean(run_i["SendExternalSpike_time"].loc[max_st_idx]) +
            np.mean(run_i["SendSpikeToRemote_time"].loc[max_st_idx]) +
            np.mean(run_i["RecvSpikeFromRemote_time"].loc[max_st_idx]) +
            np.mean(run_i["ExternalSpikeReset_time"].loc[max_st_idx]))
        local_spike_handling_delivery = (np.mean(run_i["GetSpike_time"].loc[max_st_idx]) +
            np.mean(run_i["NestedLoop_time"].loc[max_st_idx]) +
            np.mean(run_i["SpikeBufferUpdate_time"].loc[max_st_idx]) +
            np.mean(run_i["SpikeReset_time"].loc[max_st_idx]))
        oth = (np.mean(run_i["Simulation time"].loc[max_st_idx]) - 
            np.sum(np.mean(run_i.drop(columns=["Building time", "Simulation time"]).loc[max_st_idx])))
        neuron.append(np.mean(run_i["neuron_Update_time"].loc[max_st_idx]))
        poisson.append(np.mean(run_i["poisson_generator_time"].loc[max_st_idx]))
        remote_spike.append(remote_spike_handling_delivery)
        local_spike.append(local_spike_handling_delivery)
        other.append(oth)

print("\n\nAveraged results")
print("Building time [s]     :", np.mean(bt), "+/-", np.std(bt))
print("Simulation time [s]   :", np.mean(st), "+/-", np.std(st))
print("Neuron dynamics [s]   :", np.mean(neuron), "+/-", np.std(neuron))
print("Poisson generator [s] :", np.mean(poisson), "+/-", np.std(poisson))
print("Remote spike (MPI) [s]:", np.mean(remote_spike), "+/-", np.std(remote_spike))
print("Local spike [s]       :", np.mean(local_spike), "+/-", np.std(local_spike))
print("Other [s]             :", np.mean(other), "+/-", np.std(other))

mean = [np.mean(bt), np.mean(st), np.mean(neuron), np.mean(poisson), np.mean(remote_spike), np.mean(local_spike), np.mean(other)]
std = [np.std(bt), np.std(st), np.std(neuron), np.std(poisson), np.std(remote_spike), np.std(local_spike), np.std(other)]

stat ={"mean": mean,"std": std}
df_stat = pd.DataFrame(stat, index = ["Building time ", "Simulation time", "Neuron dynamics", "Poisson generator",
        "Remote spike", "Local spike", "Other"])
df_stat.to_csv("ngpu_times_stat.csv")




