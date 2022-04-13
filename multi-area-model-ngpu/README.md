# Multi-Area Model simulation with NEST GPU
This directory enables the simulation of 8 downscaled areas of the Multi-Area Model (MAM) using NEST GPU.
In this configuration, only the first 8 areas are simulated, whereas the other areas are replaced with Poisson spike generators. The simulation requires 8 nodes, with 1 MPI process each and a GPU per MPI process is needed to simulate the model appropriately.

## How to run the model
The model can be run with spike recording enabled or disabled. In the directory we provide SLURM files to run the model in a cluster. The current version supports the most recent spike recording implementation of the library. For this reason, the script ``simulation.py`` has been slightly modified. An older version of the script can be found as ``simulation_old.py``.
### Spike recording enabled
Check the model parameters and simulation setting in the file ``run_simulation.templ``. Then edit the file ``run_sbatch.sh`` to set all the details needed to correctly run the siulation (e.g. account name, path to output and error files).
For a single simulation just type

```bash
   sbatch run_sbatch.sh 
```

You can also edit the seed for random number generation by typing

```bash
   sbatch run_sbatch.sh <number>
```
This script procude first the Python script ``run_simulation.py`` and then runs it to perform the simulation.
If you want to run 10 simulations recursively you can type

```bash
   . run.sh
```

### Spike recording disabled
To run a simulation without spike recording the instructions are the same as before, but you should use different files.
The simulation parameters and setting are stored in the file ``run_eval_time.templ``. Then you should edit and run the slurm script ``run_sbatch_eval_time.sh`` for a single simulation or ``run_eval_time.sh`` for a sequence of 10 simulation with different seeds for random number generation.
