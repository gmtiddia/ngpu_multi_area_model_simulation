
"""
multiarea_model
==============

Simulation class of the multi-area model of macaque visual vortex by
Schmidt et al. (2018).


Classes
-------
Simulation : Loads a parameter file that specifies simulation
parameters for a simulation of the instance of the model. A simulation
is identified by a unique hash label.

"""

from mpi4py import MPI
import json
import nestgpu as ngpu
import numpy as np
import os
import pprint
import shutil
import time
import scipy.stats

from .analysis_helpers import _load_npy_to_dict, model_iter
from config import base_path, data_path
from copy import deepcopy
from .default_params import nested_update, sim_params
from .default_params import check_custom_params
from dicthash import dicthash
from .multiarea_helpers import extract_area_dict, create_vector_mask
try:
    from .sumatra_helpers import register_runtime
    sumatra_found = True
except ImportError:
    sumatra_found = False


class Simulation:
    def __init__(self, network, sim_spec):
        """
        Simulation class.
        An instance of the simulation class with the given parameters.
        Can be created as a member class of a multiarea_model instance
        or standalone.

        Parameters
        ----------
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network to be simulated.
        params : dict
            custom simulation parameters that overwrite the
            default parameters defined in default_params.py
        """
        ngpu.ConnectMpiInit()
        self.params = deepcopy(sim_params)
        if isinstance(sim_spec, dict):
            check_custom_params(sim_spec, self.params)
            self.custom_params = sim_spec
        else:
            fn = os.path.join(data_path,
                              sim_spec,
                              '_'.join(('custom_params',
                                        sim_spec)))
            with open(fn, 'r') as f:
                self.custom_params = json.load(f)['sim_params']

        nested_update(self.params, self.custom_params)

        self.network = network
        self.label = dicthash.generate_hash_from_dict({'params': self.params,
                                                       'network_label': self.network.label})

        print("Simulation label: {}".format(self.label))
        self.data_dir = os.path.join(data_path, self.label)
        try:
            os.mkdir(self.data_dir)
            os.mkdir(os.path.join(self.data_dir, 'recordings'))
        except OSError:
            pass
        self.copy_files()
        print("Copied files.")
        d = {'sim_params': self.custom_params,
             'network_params': self.network.custom_params,
             'network_label': self.network.label}
        with open(os.path.join(self.data_dir,
                               '_'.join(('custom_params', self.label))), 'w') as f:
            json.dump(d, f)
        print("Initialized simulation class.")

        self.areas_simulated = self.params['areas_simulated']
        self.areas_recorded = self.params['recording_dict']['areas_recorded']
        self.T = self.params['t_sim']

    def __eq__(self, other):
        # Two simulations are equal if the simulation parameters and
        # the simulated networks are equal.
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        s = "Simulation {} of network {} with parameters:".format(self.label, self.network.label)
        s += pprint.pformat(self.params, width=1)
        return s

    def copy_files(self):
        """
        Copy all relevant files for the simulation to its data directory.
        """
        files = [os.path.join('multiarea_model',
                              'data_multiarea',
                              'Model.py'),
                 os.path.join('multiarea_model',
                              'data_multiarea',
                              'VisualCortex_Data.py'),
                 os.path.join('multiarea_model',
                              'multiarea_model.py'),
                 os.path.join('multiarea_model',
                              'simulation.py'),
                 os.path.join('multiarea_model',
                              'default_params.py'),
                 os.path.join('config_files',
                              ''.join(('custom_Data_Model_', self.network.label, '.json'))),
                 os.path.join('config_files',
                              '_'.join((self.network.label, 'config')))]
        if self.network.params['connection_params']['replace_cc_input_source'] is not None:
            fs = self.network.params['connection_params']['replace_cc_input_source']
            if '.json' in fs:
                files.append(fs)
            else:  # Assume that the cc input is stored in one npy file per population
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                for it in fn_iter:
                    fp_it = (fs,) + it
                    fp_ = '{}.npy'.format('-'.join(fp_it))
                    files.append(fp_)
        for f in files:
            shutil.copy2(os.path.join(base_path, f),
                         self.data_dir)

    def prepare(self):
        """
        Prepare NEST GPU Kernel.
        """
        master_seed = self.params['master_seed']
        num_processes = self.params['num_processes']
        local_num_threads = self.params['local_num_threads']
        vp = num_processes * local_num_threads
        ngpu.SetKernelStatus({'rnd_seed': master_seed + ngpu.Rank(),
                              'max_spike_num_fact': 0.01,
                              'max_spike_per_host_fact': 0.01})
        self.pyrngs = [np.random.RandomState(s) for s in list(range(
            master_seed + vp + 1, master_seed + 2 * (vp + 1)))]

    def create_areas(self):
        """
        Create all areas with their populations and internal connections.
        """
        self.areas = []
        arank = 0
        for area_name in self.areas_simulated:
            a = Area(self, self.network, area_name, arank)
            self.areas.append(a)
            arank = arank + 1


    def cortico_cortical_input(self):
        """
        Create connections between areas.
        """
        replace_cc = self.network.params['connection_params']['replace_cc']
        replace_non_simulated_areas = self.network.params['connection_params'][
            'replace_non_simulated_areas']
        if self.network.params['connection_params']['replace_cc_input_source'] is None:
            replace_cc_input_source = None
        else:
            replace_cc_input_source = os.path.join(self.data_dir,
                                                   self.network.params['connection_params'][
                                                       'replace_cc_input_source'])

        if not replace_cc and set(self.areas_simulated) != set(self.network.area_list):
            if replace_non_simulated_areas == 'het_current_nonstat':
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                non_simulated_cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
            elif replace_non_simulated_areas == 'het_poisson_stat':
                fn = self.network.params['connection_params']['replace_cc_input_source']
                with open(fn, 'r') as f:
                    non_simulated_cc_input = json.load(f)
            elif replace_non_simulated_areas == 'hom_poisson_stat':
                non_simulated_cc_input = {source_area_name:
                                          {source_pop:
                                           self.network.params['input_params']['rate_ext']
                                           for source_pop in
                                           self.network.structure[source_area_name]}
                                          for source_area_name in self.network.area_list}
            else:
                raise KeyError("Please define a valid method to"
                               " replace non-simulated areas.")

        if replace_cc == 'het_current_nonstat':
            fn_iter = model_iter(mode='single', areas=self.network.area_list)
            cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
        elif replace_cc == 'het_poisson_stat':
            with open(self.network.params['connection_params'][
                    'replace_cc_input_source'], 'r') as f:
                cc_input = json.load(f)
        elif replace_cc == 'hom_poisson_stat':
            cc_input = {source_area_name:
                        {source_pop:
                         self.network.params['input_params']['rate_ext']
                         for source_pop in
                         self.network.structure[source_area_name]}
                        for source_area_name in self.network.area_list}

        # Connections between simulated areas are not replaced
        if not replace_cc:
            for target_area in self.areas:
                # Loop source area though complete list of areas
                for source_area_name in self.network.area_list:
                    if target_area.name != source_area_name:
                        # If source_area is part of the simulated network,
                        # connect it to target_area
                        if source_area_name in self.areas:
                            source_area = self.areas[self.areas.index(source_area_name)]
                            connect(self,
                                    target_area,
                                    source_area)
                            if source_area.rank == ngpu.Rank():
                                print("Connected area n. ", source_area.rank, " to area n. ", target_area.rank, flush=True)
                            #comm.barrier()
                        # Else, replace the input from source_area with the
                        # chosen method
                        else:
                            target_area.create_additional_input(replace_non_simulated_areas,
                                                                source_area_name,
                                                                non_simulated_cc_input[
                                                                    source_area_name])
        # Connections between all simulated areas are replaced
        else:
            for target_area in self.areas:
                for source_area in self.areas:
                    if source_area != target_area:
                        target_area.create_additional_input(replace_cc,
                                                            source_area.name,
                                                            cc_input[source_area.name])

    def simulate(self):
        """
        Create the network and execute simulation.
        Record wallclock time.
        """
        t0 = time.time()
        self.prepare()
        t1 = time.time()
        self.time_prepare = t1 - t0
        print("Prepared simulation in {0:.2f} seconds.".format(self.time_prepare), flush=True)

        self.create_areas()

        t2 = time.time()
        self.time_network_local = t2 - t1
        print("Created areas and internal connections in {0:.2f} seconds.".format(
            self.time_network_local))

        self.cortico_cortical_input()
        t3 = time.time()

        self.time_network_global = t3 - t2
        print("Created cortico-cortical connections in {0:.2f} seconds.".format(
            self.time_network_global))

        self.save_network_gids()

        ngpu.Calibrate()
        t3b = time.time()
        time_calibrate = t3b - t3
        print("Calibrated network in {0:.2f} seconds.".format(time_calibrate))

        if self.areas_recorded == []:
            ngpu.Simulate(500.0)
            t3c = time.time()
            print("Pre simulation time: {0:.2f} seconds.".format(t3c-t3b))
            ngpu.Simulate(self.T)
            t4 = time.time()
            self.time_simulate = t4 - t3c
            print("Simulated network in {0:.2f} seconds.".format(self.time_simulate))
            self.logging()
        else:
            for a in self.areas:                                                                      
                if a.rank==ngpu.Rank():
                    for pop in a.populations:
                        i0 = a.gids[pop][0]
                        i1 = a.gids[pop][1]
                        n_nodes = i1 - i0 + 1
                        neur = ngpu.NodeSeq(i0, n_nodes)
                        ngpu.SetRecSpikeTimesStep(neur, 500)
            
            ngpu.Simulate(500.0)
            print("Extracting recorded spike times for presimulation")
            spike_times_dict = self.get_recorded_spikes()
            for a in self.areas:                                                                      
                if a.rank==ngpu.Rank():
                    for pop in a.populations:
                        i0 = a.gids[pop][0]
                        i1 = a.gids[pop][1]
                        n_nodes = i1 - i0 + 1
                        neur = ngpu.NodeSeq(i0, n_nodes)
                        ngpu.SetRecSpikeTimesStep(neur, 2000)
                        
            t3c = time.time()
            print("Pre simulation time: {0:.2f} seconds.".format(t3c-t3b))
            
            ngpu.Simulate(self.T)
            t4 = time.time()
            self.time_simulate = t4 - t3c
            print("Simulated network in {0:.2f} seconds.".format(self.time_simulate))
            print("Extracting recorded spike times for simulation")
            spike_times_dict = self.get_recorded_spikes()
            self.write_spikes(spike_times_dict)

    def empty_spike_times_dict(self):
        """
        Return empty spike time dictionary for local area.
        """
        spike_times_dict = {}
        for a in self.areas:
            if a.rank==ngpu.Rank():
                for pop in a.populations:
                    spike_times_dict[pop] = []
        return spike_times_dict

    def get_recorded_spikes(self):
        """
        Extract recorded spike times.
        """
        spike_times_dict = {}
        for a in self.areas:
            if a.rank==ngpu.Rank():
                for pop in a.populations:
                    i0 = a.gids[pop][0]
                    i1 = a.gids[pop][1]
                    n_nodes = i1 - i0 + 1
                    print('   Extracting spikes for area:', a.name, ' population:', pop, ' neuron idx range:', i0, ' ', i1, flush=True)
                    data = []
                    spike_times_list = ngpu.GetRecSpikeTimes(ngpu.NodeSeq(i0, n_nodes))
                    for i in range(n_nodes):
                        i_neur = i0 + i
                        spike_times = spike_times_list[i]
                        for t in spike_times:
                            data.append([i_neur, t])
                    spike_times_dict[pop] = data
        return spike_times_dict

    def write_spikes(self, spike_times_dict):
        """
        Write recorded spike times to file.
        """
        print("Writing recorded spike times to file")
        for a in self.areas:
            if a.rank==ngpu.Rank():
                for pop in a.populations:
                    i0 = a.gids[pop][0]
                    i1 = a.gids[pop][1]
                    print('Writing spikes for area:', a.name, ' population:', pop, ' neuron idx range:', i0, ' ', i1)
                    data = spike_times_dict[pop]
                    arr = np.array(data)
                    fn = os.path.join(self.data_dir, 'recordings', 'spike_times_' + a.name + '_' + pop + 
                                      '.dat')
                    print('File: ', fn)
                    fmt='%d\t%.3f'
                    np.savetxt(fn, arr, fmt=fmt, header="sender time_ms",
                               comments='')


    def logging(self):
        """
        Write runtime for the MPI processes
        to file.
        """
        d = {'time_prepare': self.time_prepare,
             'time_network_local': self.time_network_local,
             'time_network_global': self.time_network_global,
             'time_simulate': self.time_simulate}

        fn = os.path.join(self.data_dir,
                          'recordings',
                          '_'.join((self.label,
                                    'logfile',
                                    str(ngpu.Rank()))))
        with open(fn, 'w') as f:
            json.dump(d, f)

    def save_network_gids(self):
        with open(os.path.join(self.data_dir,
                               'recordings',
                               'network_gids.txt'), 'w') as f:
            for area in self.areas:
                for pop in self.network.structure[area.name]:
                    f.write("{area},{pop},{g0},{g1}\n".format(area=area.name,
                                                              pop=pop,
                                                              g0=area.gids[pop][0],
                                                              g1=area.gids[pop][1]))

    def register_runtime(self):
        if sumatra_found:
            register_runtime(self.label)
        else:
            raise ImportWarning('Sumatra is not installed, the '
                                'runtime cannot be registered.')


class Area:
    def __init__(self, simulation, network, name, rank):
        """
        Area class.
        This class encapsulates a single area of the model.
        It creates all populations and the intrinsic connections between them.
        It provides an interface to allow connecting the area to other areas.

        Parameters
        ----------
        simulation : simulation
           An instance of the simulation class that specifies the
           simulation that the area is part of.
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network the area is part of.
        name : str
            Name of the area.
        """

        self.name = name
        self.simulation = simulation
        self.network = network
        self.rank =rank
        self.neuron_numbers = network.N[name]
        self.synapses = extract_area_dict(network.synapses,
                                          network.structure,
                                          self.name,
                                          self.name)
        self.W = extract_area_dict(network.W,
                                   network.structure,
                                   self.name,
                                   self.name)
        self.W_sd = extract_area_dict(network.W_sd,
                                      network.structure,
                                      self.name,
                                      self.name)
        self.populations = network.structure[name]

        self.external_synapses = {}
        for pop in self.populations:
            self.external_synapses[pop] = self.network.K[self.name][pop]['external']['external']

        self.create_populations()
        if rank==ngpu.Rank():
            print("Rank {}: created area {} with {} local nodes".format(ngpu.Rank(),
                                                                        self.name,
                                                                        self.num_local_nodes), flush=True)
            self.connect_devices()
            self.connect_populations()
            print("Created internal connections of area n. ", rank, " in mpi proc. ", ngpu.Rank(), flush=True)

    def __str__(self):
        s = "Area {} with {} neurons.".format(
            self.name, int(self.neuron_numbers['total']))
        return s

    def __eq__(self, other):
        # If other is an instance of area, it should be the exact same
        # area This opens the possibility to have multiple instance of
        # one cortical areas
        if isinstance(other, Area):
            return self.name == other.name and self.gids == other.gids
        elif isinstance(other, str):
            return self.name == other

    def create_populations(self):
        """
        Create all populations of the area.
        """
        self.gids = {}
        self.num_local_nodes = 0
        for pop in self.populations:
            n = int(self.neuron_numbers[pop])
            #print("Creating ", n, " neurons", flush=True)
            remote_neurons = ngpu.RemoteCreate(self.rank, self.network.params['neuron_params']['neuron_model'],
                                               int(self.neuron_numbers[pop]))
            neurons = remote_neurons.node_seq
            if ngpu.Rank() == self.rank:
                ngpu.SetStatus(neurons, self.network.params['neuron_params']['single_neuron_dict'])
                if self.name in self.simulation.params['recording_dict']['areas_recorded']:
                    ngpu.ActivateRecSpikeTimes(neurons, 100)
                    print("Activated spike times recording for area:", self.name, " population:", pop)  
                mask = create_vector_mask(self.network.structure, areas=[self.name], pops=[pop])
                I_e = self.network.add_DC_drive[mask][0]
                if not self.network.params['input_params']['poisson_input']:
                    K_ext = self.external_synapses[pop]
                    W_ext = self.network.W[self.name][pop]['external']['external']
                    tau_syn = self.network.params['neuron_params']['single_neuron_dict']['tau_syn']
                    DC = K_ext * W_ext * tau_syn * 1.e-3 * \
                         self.network.params['rate_ext']
                    I_e += DC
                ngpu.SetStatus(neurons, {'I_e': I_e})
                Vm = self.network.params['neuron_params']['V0_mean']
                Vstd = self.network.params['neuron_params']['V0_sd']
                Vmin = Vm - 3*Vstd
                Vmax = Vm + 3*Vstd
                E_L = self.network.params['neuron_params']['single_neuron_dict']['E_L']
                ngpu.SetStatus(neurons, 'V_m_rel', {"distribution":"normal_clipped", "mu":Vm-E_L, "low":Vmin-E_L,
                                                "high":Vmax-E_L, "sigma":Vstd})
                self.num_local_nodes += len(neurons)

            # Store first and last GID of each population
            self.gids[pop] = (neurons[0], neurons[-1])

    def connect_populations(self):
        """
        Create connections between populations.
        """
        connect(self.simulation,
                self,
                self)

    def connect_devices(self):
        # TO BE DONE
        #if self.simulation.params['recording_dict']['record_vm']:
        #    for pop in self.populations:
        #        nrec = int(self.simulation.params['recording_dict']['Nrec_vm_fraction'] *
        #                   self.neuron_numbers[pop])
        #        nest.Connect(self.simulation.voltmeter,
        #                     tuple(range(self.gids[pop][0], self.gids[pop][0] + nrec + 1)))
        if self.network.params['input_params']['poisson_input']:
            self.poisson_generators = []
            for pop in self.populations:
                K_ext = self.external_synapses[pop]
                W_ext = self.network.W[self.name][pop]['external']['external']
                pg = ngpu.Create('poisson_generator', 1)
                print('Created 1 poisson generator for area n. ', self.rank, ' population:', pop, flush=True)
                ngpu.SetStatus(
                    pg, {'rate': self.network.params['input_params']['rate_ext'] * K_ext})
                conn_spec = {'rule': 'all_to_all'}
                syn_spec = {'weight': W_ext, 'delay': 0.1}
                i0 = self.gids[pop][0]
                n = self.gids[pop][1] - i0 + 1
                ngpu.Connect(pg, ngpu.NodeSeq(i0, n), conn_spec, syn_spec)
                self.poisson_generators.append(pg[0])

    def create_additional_input(self, input_type, source_area_name, cc_input):
        """
        TO BE DONE
        Replace the input from a source area by the chosen type of input.

        Parameters
        ----------
        input_type : str, {'het_current_nonstat', 'hom_poisson_stat',
                           'het_poisson_stat'}
            Type of input to replace source area. The source area can
            be replaced by Poisson sources with the same global rate
            rate_ext ('hom_poisson_stat') or by specific rates
            ('het_poisson_stat') or by time-varying specific current
            ('het_current_nonstat')
        source_area_name: str
            Name of the source area to be replaced.
        cc_input : dict
            Dictionary of cortico-cortical input of the process
            replacing the source area.
        """


def connect(simulation,
            target_area,
            source_area):
    """
    Connect two areas with each other.

    Parameters
    ----------
    simulation : Simulation instance
        Simulation simulating the network containing the two areas.
    target_area : Area instance
        Target area of the projection
    source_area : Area instance
        Source area of the projection
    """
    network = simulation.network
    synapses = extract_area_dict(network.synapses,
                                 network.structure,
                                 target_area.name,
                                 source_area.name)
    W = extract_area_dict(network.W,
                          network.structure,
                          target_area.name,
                          source_area.name)
    W_sd = extract_area_dict(network.W_sd,
                             network.structure,
                             target_area.name,
                             source_area.name)

    for target in target_area.populations:
        for source in source_area.populations:
            conn_spec = {'rule': 'fixed_total_number',
                         'total_num': int(synapses[target][source])}

            syn_weight = {'distribution': 'normal_clipped',
                          'mu': W[target][source],
                          'sigma': W_sd[target][source]}
            if target_area == source_area:
                if 'E' in source:
                    syn_weight.update({'low': 0.})
                    mean_delay = network.params['delay_params']['delay_e']
                elif 'I' in source:
                    syn_weight.update({'high': 0.})
                    mean_delay = network.params['delay_params']['delay_i']
                std_delay = mean_delay * network.params['delay_params']['delay_rel']
                min_delay = simulation.params['dt']
                max_delay = mean_delay + 2.0 * std_delay
            else:
                v = network.params['delay_params']['interarea_speed']
                s = network.distances[target_area.name][source_area.name]
                mean_delay = s / v
                std_delay = mean_delay * network.params['delay_params']['delay_rel']
                min_delay = simulation.params['dt'] # mean_delay - 1.0 * std_delay
                max_delay = mean_delay + 2.0 * std_delay

            #delay_nparray = scipy.stats.truncnorm.rvs(
            #    (min_delay-mean_delay)/std_delay,
            #    (max_delay-mean_delay)/std_delay,
            #    loc=mean_delay, scale=std_delay, size=int(synapses[target][source]))
            sd = 2.0*std_delay / 30
            rsd = max(round(sd, 1), 0.1)
            #delay_list = [max(rsd*round(x/rsd), 0.1) for x in delay_nparray]
            #syn_delay = {"array": delay_list}
            syn_delay = {'distribution': 'normal_clipped',
                         'low': min_delay,
                         'high': max_delay,
                         'mu': mean_delay,
                         'sigma': std_delay,
                         'step': rsd } 
            syn_spec = {'weight': syn_weight,
                        'delay': syn_delay}

            source_i0 = source_area.gids[source][0]
            source_n = source_area.gids[source][1] - source_i0 + 1
            target_i0 = target_area.gids[target][0]
            target_n = target_area.gids[target][1] - target_i0 + 1
            ngpu.RemoteConnect(source_area.rank, ngpu.NodeSeq(source_i0, source_n),
                               target_area.rank, ngpu.NodeSeq(target_i0, target_n),
                               conn_spec,
                               syn_spec)
