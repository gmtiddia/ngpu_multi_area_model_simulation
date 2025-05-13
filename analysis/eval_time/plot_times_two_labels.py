import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cifre=17+5
titolo=18+5
label=25+5

#read ms data from csv
ngpu_gs = pd.read_csv('ground_state/ngpu_times_stat.csv')
print("NEST GPU ground state")
print(ngpu_gs)
ngpu_gs_building = ngpu_gs.iloc[0]['mean']
ngpu_gs_simulation = ngpu_gs.iloc[1]['mean']/10.0
ngpu_gs_simulation_std = ngpu_gs.iloc[1]['std']/10.0
ngpu_gs_neuron_sim = ngpu_gs.iloc[2]['mean']/10.0
ngpu_gs_poisson = ngpu_gs.iloc[3]['mean']/10.0
ngpu_gs_update = ngpu_gs_neuron_sim + ngpu_gs_poisson
#ngpu_gs_remote_spike_handling = ngpu_gs.iloc[4]['mean']/10.0
ngpu_gs_remote_spike_handling_dum = ngpu_gs.iloc[4]['mean']/10.0
coll_frac_gs = 0.4
ngpu_gs_collocation = ngpu_gs_remote_spike_handling_dum*coll_frac_gs
ngpu_gs_remote_spike_handling = ngpu_gs_remote_spike_handling_dum - ngpu_gs_collocation
ngpu_gs_local_spike_handling = ngpu_gs.iloc[5]['mean']/10.0
ngpu_gs_other = ngpu_gs.iloc[6]['mean']/10.0

#nest ms data [only 32 nodes data]
nest_gs = pd.read_csv('ground_state/nest_times_stat.csv')
print("NEST ground state")
print(nest_gs)
nest_gs_building = nest_gs.iloc[0]['mean']
nest_gs_simulation = nest_gs.iloc[1]['mean']/10.0
nest_gs_simulation_std = nest_gs.iloc[1]['std']/10.0
nest_gs_update = nest_gs.iloc[2]['mean']/10.0
nest_gs_collocation = nest_gs.iloc[3]['mean']/10.0
nest_gs_communication = nest_gs.iloc[4]['mean']/10.0
nest_gs_delivery = nest_gs.iloc[5]['mean']/10.0
nest_gs_other = nest_gs_simulation - (nest_gs_delivery + nest_gs_communication + nest_gs_collocation + nest_gs_update)
print(nest_gs_simulation - (nest_gs_delivery + nest_gs_communication + nest_gs_collocation + nest_gs_update))


#read ms data from csv
ngpu_ms = pd.read_csv('metastable_state/ngpu_times_stat.csv')
print("NEST GPU metastable state")
print(ngpu_ms)
ngpu_ms_building = ngpu_ms.iloc[0]['mean']
ngpu_ms_simulation = ngpu_ms.iloc[1]['mean']/10.0
ngpu_ms_simulation_std = ngpu_ms.iloc[1]['std']/10.0
ngpu_ms_neuron_sim = ngpu_ms.iloc[2]['mean']/10.0
ngpu_ms_poisson = ngpu_ms.iloc[3]['mean']/10.0
ngpu_ms_update = ngpu_ms_neuron_sim + ngpu_ms_poisson
#ngpu_ms_remote_spike_handling = ngpu_ms.iloc[4]['mean']/10.0
ngpu_ms_remote_spike_handling_dum = ngpu_ms.iloc[4]['mean']/10.0
coll_frac_ms = 0.09
ngpu_ms_collocation = ngpu_ms_remote_spike_handling_dum*coll_frac_ms
ngpu_ms_remote_spike_handling = ngpu_ms_remote_spike_handling_dum - ngpu_ms_collocation
ngpu_ms_local_spike_handling = ngpu_ms.iloc[5]['mean']/10.0
ngpu_ms_other = ngpu_ms.iloc[6]['mean']/10.0

#nest ms data [only 32 nodes data]
nest_ms = pd.read_csv('metastable_state/nest_times_stat.csv')
print("NEST metastable state")
print(nest_ms)
nest_ms_building = nest_ms.iloc[0]['mean']
nest_ms_simulation = nest_ms.iloc[1]['mean']/10.0
nest_ms_simulation_std = nest_ms.iloc[1]['std']/10.0
nest_ms_update = nest_ms.iloc[2]['mean']/10.0
nest_ms_collocation = nest_ms.iloc[3]['mean']/10.0
nest_ms_communication = nest_ms.iloc[4]['mean']/10.0
nest_ms_delivery = nest_ms.iloc[5]['mean']/10.0
nest_ms_other = nest_ms_simulation - (nest_ms_delivery + nest_ms_communication + nest_ms_collocation + nest_ms_update)
print(nest_ms_simulation - (nest_ms_delivery + nest_ms_communication + nest_ms_collocation + nest_ms_update))

names=['NEST GPU', 'NEST', ' NEST GPU ', ' NEST ']
y=np.arange(len(names))
fig, ax = plt.subplots(1,2, figsize=(32,20))
#plot2: plot unico dove le barre si sommano
xnest_ms=[nest_ms_other, nest_ms_update, nest_ms_collocation, nest_ms_communication, nest_ms_delivery]
colorsnest=['brown', '#ee8866','#eedd88', '#44bb99', '#77aadd']
xngpu_ms=[ngpu_ms_other, ngpu_ms_update, ngpu_ms_collocation, ngpu_ms_remote_spike_handling, ngpu_ms_local_spike_handling]

#plot2: plot unico dove le barre si sommano
xnest_gs=[nest_gs_other, nest_gs_update, nest_gs_collocation, nest_gs_communication, nest_gs_delivery]
colorsnest=['brown', '#ee8866', '#eedd88', '#44bb99', '#77aadd']
xngpu_gs=[ngpu_gs_other, ngpu_gs_update, ngpu_gs_collocation, ngpu_gs_remote_spike_handling, ngpu_gs_local_spike_handling]

colorsngpu=['brown', '#ee8866', '#eedd88', '#44bb99', '#77aadd']

ngpu1_gs, = ax[0].bar(names[0], xngpu_gs[4], bottom=np.sum(xngpu_gs[0:4]), color=colorsngpu[4], edgecolor="k", linewidth = 0.5)
ngpu2_gs, = ax[0].bar(names[0], xngpu_gs[3], bottom=np.sum(xngpu_gs[0:3]), color=colorsngpu[3], edgecolor="k", linewidth = 0.5)
ngpu3_gs, = ax[0].bar(names[0], xngpu_gs[2], bottom=np.sum(xngpu_gs[0:2]), color=colorsngpu[2], edgecolor="k", linewidth = 0.5)
ngpu4_gs, = ax[0].bar(names[0], xngpu_gs[1], bottom=xngpu_gs[0], color=colorsngpu[1], edgecolor="k", linewidth = 0.5)
ngpu5_gs, = ax[0].bar(names[0], xngpu_gs[0], color=colorsngpu[0], edgecolor="k", linewidth = 0.5)
ngpu0_gs, = ax[0].bar(names[0], ngpu_gs_simulation, color='none', yerr=ngpu_gs_simulation_std, capsize=5, error_kw={"elinewidth": 2.25})

nest1_gs, = ax[0].bar(names[1], xnest_gs[4], bottom=np.sum(xnest_gs[0:4]), color=colorsnest[4], edgecolor="k", linewidth = 0.5)
nest2_gs, = ax[0].bar(names[1], xnest_gs[3], bottom=np.sum(xnest_gs[0:3]), color=colorsnest[3], edgecolor="k", linewidth = 0.5)
nest3_gs, = ax[0].bar(names[1], xnest_gs[2], bottom=np.sum(xnest_gs[0:2]), color=colorsnest[2], edgecolor="k", linewidth = 0.5)
nest4_gs, = ax[0].bar(names[1], xnest_gs[1], bottom=xnest_gs[0], color=colorsnest[1], edgecolor="k", linewidth = 0.5)
nest5_gs, = ax[0].bar(names[1], xnest_gs[0], color=colorsnest[0], edgecolor="k", linewidth=0.5)
nest0_gs, = ax[0].bar(names[1], nest_gs_simulation, color='none', yerr=nest_gs_simulation_std, capsize=5, error_kw={"elinewidth": 2.25})


#plot ngpu
#dovrebbe essere possibile usare il comando bottom anzichè sommare i dati, fare delle prove a tempo debito per avere maggior rigore
ngpu1_ms, = ax[0].bar(names[2], xngpu_ms[4], bottom=np.sum(xngpu_ms[0:4]), color=colorsngpu[4], edgecolor="k", linewidth = 0.5)
ngpu2_ms, = ax[0].bar(names[2], xngpu_ms[3], bottom=np.sum(xngpu_ms[0:3]), color=colorsngpu[3], edgecolor="k", linewidth = 0.5)
ngpu3_ms, = ax[0].bar(names[2], xngpu_ms[2], bottom=np.sum(xngpu_ms[0:2]), color=colorsngpu[2], edgecolor="k", linewidth = 0.5)
ngpu4_ms, = ax[0].bar(names[2], xngpu_ms[1], bottom=xngpu_ms[0], color=colorsngpu[1], edgecolor="k", linewidth = 0.5)
ngpu5_ms, = ax[0].bar(names[2], xngpu_ms[0], color=colorsngpu[0], edgecolor="k", linewidth = 0.5)
ngpu0_ms, = ax[0].bar(names[2], ngpu_ms_simulation, color='none', yerr=ngpu_ms_simulation_std, capsize=5, error_kw={"elinewidth": 2.25})

nest1_ms, = ax[0].bar(names[3], xnest_ms[4], bottom=np.sum(xnest_ms[0:4]), color=colorsnest[4], edgecolor="k", linewidth = 0.5)
nest2_ms, = ax[0].bar(names[3], xnest_ms[3], bottom=np.sum(xnest_ms[0:3]), color=colorsnest[3], edgecolor="k", linewidth = 0.5)
nest3_ms, = ax[0].bar(names[3], xnest_ms[2], bottom=np.sum(xnest_ms[0:2]), color=colorsnest[2], edgecolor="k", linewidth = 0.5)
nest4_ms, = ax[0].bar(names[3], xnest_ms[1], bottom=xnest_ms[0], color=colorsnest[1], edgecolor="k", linewidth = 0.5)
nest5_ms, = ax[0].bar(names[3], xnest_ms[0], color=colorsnest[0], edgecolor="k", linewidth=0.5)
nest0_ms, = ax[0].bar(names[3], nest_ms_simulation, color='none', yerr=nest_ms_simulation_std, capsize=5, error_kw={"elinewidth": 2.25})


plt.rcParams['hatch.linewidth'] = 3



#making the legend only for those barplots

#second_legend= plt.legend([nest1_ms, nest2_ms, nest3_ms, nest4_ms, nest5_ms], ['Delivery', 'Communication', 'Collocation', 'Update', 'Other'], prop={'size': titolo}, bbox_to_anchor=(0.5, 1))
#ax = plt.gca().add_artist(second_legend)
#first_legend.set_title('NEST GPU (JUSUF)', prop={'size': titolo+2})
#second_legend.set_title('NEST (JURECA-DC)', prop={'size': titolo+2})
ax[0].axhline(y=1, color='k', linestyle='--')
ax[0].axvline(x=1.5, color='silver', linestyle='-')
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
ax[0].set_ylabel(r'$\mathit{T}_{ wall }\;/\;\mathit{T}_{ model }$ ',fontsize=cifre+4)
ax[0].text(0.5,-5.5,'ground state', fontsize=cifre+7, horizontalalignment="center")
ax[0].text(2.5,-5.5,'metastable state', fontsize=cifre+7, horizontalalignment="center")
ax[0].tick_params(labelsize=cifre+1)
ax[0].tick_params(axis='y',labelsize=cifre+4)
#fig.set_size_inches(32, 18)
#plt.savefig("sim_time_NESTGPU.png", format='png')
ax[0].legend([ngpu1_ms, ngpu2_ms, ngpu3_ms, ngpu4_ms, ngpu5_ms], ['Delivery', 'Communication', 'Collocation', 'Update', 'Other'], prop={'size': titolo+2}, loc='upper left')
#ax[0] = plt.gca().add_artist(first_legend)
ax[0].text(-0.085,1.0,"A", transform=ax[0].transAxes, weight="bold", fontsize=label)

df_area = pd.read_csv("metastable_state/partial_ms.csv")



df_area.plot(ax=ax[1], x="Name", kind = 'bar', stacked=True, color = ['brown', '#ee8866', '#eedd88', '#44bb99', '#77aadd'], linewidth = 0.5, edgecolor="k", legend=False)

ax[1].set_xlabel("Area",fontsize=cifre+7)
ax[1].set_ylabel(r'$\mathit{T}_{ wall }\;[\%]$ ',fontsize=cifre+4)
ax[1].tick_params(labelsize=cifre+1)
ax[1].tick_params(axis='y',labelsize=cifre+4)
ax[1].tick_params(axis='x', rotation=70)
#handles, labels = plt.gca().get_legend_handles_labels()
#plt.legend(handles[::-1], labels[::-1], prop={'size': titolo}, ncol=5,loc='upper right')
ax[1].set_ylim(0,102.0)
ax[1].text(-0.085,1.0,"B", transform=ax[1].transAxes, weight="bold", fontsize=label)
plt.subplots_adjust(left=0.045, right=0.99, top=0.96, bottom=0.11, wspace=0.15)

plt.savefig("figure8.png")

plt.show()
