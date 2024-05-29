"""
This script is used to run a simulation

It initializes the network class and then runs the simulate method of
the simulation class instance.

"""

import json
import os
import sys
import numpy as np
from config import base_path, data_path
from multiarea_model import MultiAreaModel
from mpi4py import MPI
import shutil
from multiarea_model.default_params import nested_update, sim_params

import nestgpu as ngpu

ngpu.ConnectMpiInit()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


mpi_np = ngpu.HostNum()

print(mpi_np)
print(ngpu.HostId())

d = {}
conn_params = {'g': -11.,
               'K_stable': os.path.join(base_path, 'K_stable.npy'),
               'replace_non_simulated_areas': 'hom_poisson_stat',
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.,
	       'cc_weights_factor': 1.9,
	       'cc_weights_I_factor': 2.0}
input_params = {'rate_ext': 10.}

neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}

fn = os.path.join(base_path, 'tests/fullscale_rates.json')
network_params = {'N_scaling': 0.01,
                  'K_scaling': 0.01,
		  'fullscale_rates': fn,
                  'connection_params': conn_params,
                  'input_params': input_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': 10000.,
              'areas_simulated': ['V1', 'V2', 'VP', 'V3'],
              'num_processes': 4,
              'local_num_threads': 64,
              'recording_dict': {'record_vm': False, 'areas_recorded':[]}}

theory_params = {'dt': 0.1}

if rank==0:
    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=True,
                       theory_spec=theory_params)
    p, r = M.theory.integrate_siegert()
    print("Mean-field theory predicts an average "
          "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))

    sim_params['master_seed'] = 12345
    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params)
    label = M.simulation.label
    # Copy run_simulation script to simulation folder
    shutil.copy2(os.path.join(base_path, 'run_eval_time.py'),
                 os.path.join(data_path, label))

    # Load simulation parameters
    fn = os.path.join(data_path,
                      label,
                      '_'.join(('custom_params',
                                label)))
    with open(fn, 'r') as f:
        custom_params = json.load(f)
    nested_update(sim_params, custom_params['sim_params'])

    # Copy custom param file for each MPI process
    for i in range(sim_params['num_processes']):
        shutil.copy(fn, '_'.join((fn, str(i))))
    # Collect relevant arguments for job script
    num_vp = sim_params['num_processes'] * sim_params[
        'local_num_threads']
    d = {'label': label,
         'network_label': custom_params['network_label'],
         'base_path': base_path,
         'sim_dir': os.path.join(data_path, label),
         'local_num_threads': sim_params['local_num_threads'],
         'num_processes': sim_params['num_processes'],
         'num_vp': num_vp}

else:
    label = None

label = comm.bcast(label, root=0)

fn = os.path.join(data_path,
                  label,
                  '_'.join(('custom_params',
                            label,
                           str(rank))))
with open(fn, 'r') as f:
    custom_params = json.load(f)

os.remove(fn)

network_label = custom_params['network_label']

M = MultiAreaModel(network_label,
                   simulation=True,
                   analysis=False,
                   sim_spec=custom_params['sim_params'])
print("IMHERE")
M.simulation.simulate()
