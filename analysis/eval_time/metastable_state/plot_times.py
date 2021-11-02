import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cifre=17
titolo=18

#read data from csv
ngpu = pd.read_csv('ngpu_times_stat.csv')
print(ngpu)
ngpu_building = ngpu.iloc[0]['mean']
ngpu_simulation = ngpu.iloc[1]['mean']/10.0
ngpu_simulation_std = ngpu.iloc[1]['std']/10.0
ngpu_neuron_sim = ngpu.iloc[2]['mean']/10.0
ngpu_poisson = ngpu.iloc[3]['mean']/10.0
ngpu_remote_spike_handling = ngpu.iloc[4]['mean']/10.0
ngpu_local_spike_handling = ngpu.iloc[5]['mean']/10.0
ngpu_other = ngpu.iloc[6]['mean']/10.0

#nest data [only 32 nodes data]
nest = pd.read_csv('nest_times_stat.csv')
print(nest)
nest_building = nest.iloc[0]['mean']
nest_simulation = nest.iloc[1]['mean']/10.0
nest_simulation_std = nest.iloc[1]['std']/10.0
nest_update = nest.iloc[2]['mean']/10.0
nest_collocation = nest.iloc[3]['mean']/10.0
nest_communication = nest.iloc[4]['mean']/10.0
nest_delivery = nest.iloc[5]['mean']/10.0
print(nest_simulation - (nest_delivery + nest_communication + nest_collocation + nest_update))

names=['NeuronGPU', 'NEST']
y=np.arange(len(names))
fig=plt.figure(1)
xnest=[nest_update, nest_communication, nest_delivery, nest_collocation]
colorsnest=['orange', 'limegreen', 'royalblue', 'gold']
xngpu=[ngpu_poisson, ngpu_neuron_sim, ngpu_local_spike_handling, ngpu_remote_spike_handling, ngpu_other]
colorsngpu=[ 'tomato', 'orange', 'limegreen', 'royalblue', 'gold']
plt.figure(1)

ngpu1, = plt.bar(names[0], xngpu[4], bottom=np.sum(xngpu[0:4]), color=colorsngpu[4])
ngpu2, = plt.bar(names[0], xngpu[3], bottom=np.sum(xngpu[0:3]), color=colorsngpu[3])
ngpu3, = plt.bar(names[0], xngpu[2], bottom=np.sum(xngpu[0:2]), color=colorsngpu[2])
ngpu4, = plt.bar(names[0], xngpu[1], bottom=xngpu[0], color=colorsngpu[1])
ngpu5, = plt.bar(names[0], xngpu[0], color=colorsngpu[0])
ngpu0, = plt.bar(names[0], ngpu_simulation, color='none', yerr=ngpu_simulation_std, capsize=5)

nest1, = plt.bar(names[1], xnest[3], bottom=np.sum(xnest[0:3]), color=colorsnest[3])
nest2, = plt.bar(names[1], xnest[2], bottom=np.sum(xnest[0:2]), color=colorsnest[2])
nest3, = plt.bar(names[1], xnest[1], bottom=xnest[0], color=colorsnest[1])
nest4, = plt.bar(names[1], xnest[0], color=colorsnest[0], edgecolor='tomato', hatch="//", linewidth=0)
nest0, = plt.bar(names[1], nest_simulation, color='none', yerr=nest_simulation_std, capsize=5)
plt.rcParams['hatch.linewidth'] = 3

first_legend= plt.legend([ngpu1, ngpu2, ngpu3, ngpu4, ngpu5], ['Other', 'Local spikes handling \nand delivery', 'Remote spikes handling \nand delivery (MPI)', 'Neuron dynamics', 'Poisson generators'], prop={'size': titolo}, loc='upper left')
ax = plt.gca().add_artist(first_legend)
second_legend= plt.legend([nest1, nest2, nest3, nest4], ['Collocation', 'Delivery', 'Communication', 'Update'], prop={'size': titolo}, bbox_to_anchor=(0.55, 1))
ax = plt.gca().add_artist(second_legend)
first_legend.set_title('NeuronGPU', prop={'size': titolo+2})
second_legend.set_title('NEST', prop={'size': titolo+2})
plt.axhline(y=1, color='k', linestyle='--')
plt.ylabel('Simulation time [s]',fontsize=cifre+2)
plt.tick_params(labelsize=cifre+2)
fig.set_size_inches(32, 18)
plt.savefig("sim_time_ms.png", format='png')
plt.show()
