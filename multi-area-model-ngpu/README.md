# NeuronGPU implementation of the Multi-Area Model

In ``config.py`` is contained a template to submit jobs on a cluster with Slurm. To run the model in a local machine could be used OpenMP and MPI.

Running
```
run.sh
```
launches 10 simulations (with spike recording) with different seeds for random number generations, whereas
```
run_eval_time.sh
```
launches 10 simulations without spike recording. In particular those scripts use the files ``run_simulation.templ`` and ``run_eval_time.templ`` to generate the homonymous Python scripts. The simulation parameters could be modified by editing the .templ files.
Running
```
create_symbolic_links.sh
```
in the folder in which the simulations spike times are stored create the folders data0 - data9. Those folders contains in the subfolder ``spikes_pop_idx`` the spike times of each of the 254 populations of the model stored in spike_times_i.dat, where i goes from 0 to 253.
