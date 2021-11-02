import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("metastable_NEST.csv")
df = df[df['num_nodes']==32]
names = ['Building time', 'Simulation time', 'Update', 'Collocation', 'Communication', 'Delivery']

bt = df['py_time_create'] + df['py_time_connect']
st = df['wall_time_sim']
update= df['wall_time_phase_update']
collocation= df['wall_time_phase_collocate']
communication= df['wall_time_phase_communicate']
delivery= df['wall_time_phase_deliver']

print("\n\nAveraged results")
print("Building time [s]       :", np.mean(bt), "+/-", np.std(bt))
print("Simulation time [s]     :", np.mean(st), "+/-", np.std(st))
print("Dynamics update [s]     :", np.mean(update), "+/-", np.std(update))
print("Communications (MPI) [s]:", np.mean(communication), "+/-", np.std(communication))
print("Delivery [s]            :", np.mean(delivery), "+/-", np.std(delivery))
print("Collocation [s]         :", np.mean(collocation), "+/-", np.std(collocation))

mean = [np.mean(bt), np.mean(st), np.mean(update), np.mean(collocation), np.mean(communication), np.mean(delivery)]
std = [np.std(bt), np.std(st), np.std(update), np.std(collocation), np.std(communication), np.std(delivery)]

stat ={"mean": mean,"std": std}
df_stat = pd.DataFrame(stat, index = names)
df_stat.to_csv("nest_times_stat.csv")




