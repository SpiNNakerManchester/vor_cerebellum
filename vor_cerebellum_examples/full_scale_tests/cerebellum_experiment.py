# Copyright (c) 2019-2022 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pyNN.spiNNaker as sim

from pyNN.random import RandomDistribution, NumpyRNG

# PAB imports
import traceback
import neo
import numpy as np
import os
from datetime import datetime
# general parameters
from vor_cerebellum.parameters import (CONNECTIVITY_MAP, rbls, neuron_params)
# MF-VN params
from vor_cerebellum.parameters import (mfvn_min_weight, mfvn_max_weight,
                                       mfvn_initial_weight,
                                       mfvn_ltp_constant,
                                       mfvn_beta, mfvn_sigma,
                                       vn_neuron_params, mfvn_ltd_constant)
# PF-PC params
from vor_cerebellum.parameters import (pfpc_min_weight, pfpc_max_weight,
                                       pfpc_initial_weight,
                                       pfpc_ltp_constant, pfpc_t_peak,
                                       pfpc_ltd_constant,
                                       pc_neuron_params)
from vor_cerebellum.provenance_analysis import (
    provenance_analysis, save_provenance_to_file_from_database)
from vor_cerebellum.utilities import (
    sensorial_activity, ff_1_to_1_odd_even_mapping_reversed,
    generate_head_position_and_velocity, POS_TO_VEL, ICUB_VOR_VENV_POP_SIZE,
    enable_recordings_for, result_dir, take_connectivity_snapshot,
    retrieve_all_spikes, retrieve_all_other_recordings,
    retrieve_and_package_results, get_plot_order, convert_spikes,
    remap_odd_even, remap_second_half_descending, retrieve_git_commit,
    analyse_run, fig_folder)
# Imports for SpiNNGym env
import spinn_gym as gym
from vor_cerebellum.vor_argparser import args

# Record SCRIPT start time (wall clock)
start_time = datetime.now()

# Starting to record additional parameters

USE_MOTION_TARGET = args.target_reaching  # default false

# cerebellum test bench
runtime = args.simtime  # default=10k ms
suffix = args.suffix

single_runtime = args.single_simtime  # default=10k ms
sample_time = args.error_window_size  # default 10 ms

# SpiNNGym settings
gain = args.gain
vel_to_pos = 1 / (2 * np.pi * args.slowdown_factor * gain)

# Passed-in args
mfvn_ltd_constant = args.mfvn_scale or mfvn_ltd_constant
pfpc_ltd_constant = args.pfpc_scale or pfpc_ltd_constant

mfvn_max_weight = args.mfvn_max_weight or mfvn_max_weight
pfpc_max_weight = args.pfpc_max_weight or pfpc_max_weight

mfvn_ltp_constant = args.mfvn_ltp_constant or mfvn_ltp_constant
pfpc_ltp_constant = args.pfpc_ltp_constant or pfpc_ltp_constant

vn_neuron_params['cm'] = args.vn_cm or vn_neuron_params['cm']

# I think some normalisation is in order for different slowdowns
# pfpc_ltd_constant /= args.slowdown_factor
# pfpc_ltp_constant /= args.slowdown_factor
# mfvn_ltd_constant /= args.slowdown_factor
# mfvn_ltp_constant /= args.slowdown_factor

# Synapse parameters
gc_pc_weights = 0.005
mf_vn_weights = 0.0005
# 0.08 also worked for 5x slowdown, 0.01 was the "original"
pc_vn_weights = args.pcvn_weight or 0.005
cf_pc_weights = 0.005  # 0.005
mf_gc_weights = 0.5
go_gc_weights = 0.002
mf_go_weights = 0.08

# Network parameters
num_MF_neurons = 100
num_GC_neurons = 2000
num_GOC_neurons = 100
num_PC_neurons = 200
num_VN_neurons = 200
num_CF_neurons = 200

# Random distribution for synapses delays and weights (MF and GO)
delay_distr = RandomDistribution(
    'uniform', (1.0, 10.0), rng=NumpyRNG(seed=85524))

weight_distr_MF = RandomDistribution(
    'uniform', (mf_gc_weights * 0.8, mf_gc_weights * 1.2),
    rng=NumpyRNG(seed=85524))

weight_distr_GO = RandomDistribution(
    'uniform', (go_gc_weights * 0.8, go_gc_weights * 1.2),
    rng=NumpyRNG(seed=24568))

all_neurons = {
    "mossy_fibres": num_MF_neurons,
    "granule": num_GC_neurons,
    "golgi": num_GOC_neurons,
    "purkinje": num_PC_neurons,
    "vn": num_VN_neurons,
    "climbing_fibres": num_CF_neurons
}

all_neuron_params = {
    "mossy_fibres": None,
    "granule": neuron_params,
    "golgi": neuron_params,
    "purkinje": pc_neuron_params,
    "vn": vn_neuron_params,
    "climbing_fibres": None
}

all_populations = {}

initial_connectivity = {}

final_connectivity = {}

all_projections = {}

# Weights of pf_pc
weight_dist_pfpc = RandomDistribution('uniform',
                                      (pfpc_initial_weight * 0.8,
                                       pfpc_initial_weight * 1.2),
                                      rng=NumpyRNG(seed=24534))

global_n_neurons_per_core = 100
ss_neurons_per_core = 10
pressured_npc = 10
per_pop_neurons_per_core_constraint = {
    'mossy_fibres': ss_neurons_per_core,  # global_n_neurons_per_core,
    'granule': global_n_neurons_per_core,
    'golgi': global_n_neurons_per_core,
    'purkinje': pressured_npc,
    'vn': pressured_npc,
    'climbing_fibres': 1,
}

sim.setup(timestep=1., min_delay=1, max_delay=15)
# sim.set_number_of_neurons_per_core(
#     sim.SpikeSourcePoisson, ss_neurons_per_core)
# sim.set_number_of_neurons_per_core(
#     sim.SpikeSourceArray, ss_neurons_per_core)
# sim.set_number_of_neurons_per_core(
#     sim.IF_cond_exp, global_n_neurons_per_core)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
# sim.set_number_of_neurons_per_core(
#     sim.extra_models.IFCondExpCerebellum, global_n_neurons_per_core)
# sim.set_number_of_neurons_per_core(
#     sim.extra_models.SpikeSourcePoissonVariable, ss_neurons_per_core)

# Sensorial Activity: input activity from vestibulus (will come from the head
# IMU, now it is a test bench) We simulate the output of the head encoders
# with a sinusoidal function. Each "sensorial activity" value is derived from
# the head position and velocity. From that value, we generate the mean firing
# rate of the MF neurons (later this will be an input that will come from the
# robot, through the spinnLink), the neurons that are active depend on the
# value of the sensorial activity. For each a gaussian is created centered on a
# specific neuron

# Prepare variables once at beginning
MAX_AMPLITUDE = 0.8
RELATIVE_AMPLITUDE = 1.0
_head_pos = []
_head_vel = []

i = np.arange(0, 1000, 0.001)
for t in i:
    desired_speed = -np.cos(
        t * 2 * np.pi) * MAX_AMPLITUDE * RELATIVE_AMPLITUDE * 2.0 * np.pi
    desired_pos = -np.sin(
        t * 2 * np.pi) * MAX_AMPLITUDE * RELATIVE_AMPLITUDE
    _head_pos.append(desired_pos)
    _head_vel.append(desired_speed)

normalised_head_pos = (np.asarray(_head_pos) + 0.8) / 1.6
normalised_head_vel = (
    np.asarray(_head_vel) + 0.8 * 2 * np.pi) / (1.6 * 2 * np.pi)

###############################################################################
# ============================ Create populations =========================== #
###############################################################################

# Create MF population - fake input population that will be substituted by
# external input from robot
all_mf_rates = np.ones((num_MF_neurons, runtime // sample_time)) * np.nan
all_mf_starts = np.repeat(
    [np.arange(runtime // sample_time) * sample_time], num_MF_neurons, axis=0)
all_mf_durations = np.ones(
    (num_MF_neurons, runtime // sample_time)) * sample_time
for i in np.arange(runtime // (
        sample_time * args.slowdown_factor)) * args.slowdown_factor:
    sample_no = i * sample_time // args.slowdown_factor
    current_rates = sensorial_activity(
        _head_pos[sample_no], _head_vel[sample_no])[0]
    for j in range(args.slowdown_factor):
        all_mf_rates[:, i + j] = current_rates

MF_population = sim.Population(num_MF_neurons,  # number of sources
                               sim.extra_models.SpikeSourcePoissonVariable,
                               {'rates': all_mf_rates,
                                'starts': all_mf_starts,
                                'durations': all_mf_durations
                                },  # source spike times
                               label="MF",
                               additional_parameters={'seed': 24534}
                               )

all_populations["mossy_fibres"] = MF_population

# Create GOC population
GOC_population = sim.Population(
    num_GOC_neurons, sim.IF_cond_exp(), label='GoC',
    additional_parameters={"rb_left_shifts": rbls['golgi']})
all_populations["golgi"] = GOC_population

# create PC population
PC_population = sim.Population(
    num_PC_neurons,  # number of neurons
    sim.extra_models.IFCondExpCerebellum(**pc_neuron_params),  # Neuron model
    label="PC",
    additional_parameters={"rb_left_shifts": rbls['purkinje']})
all_populations["purkinje"] = PC_population

# create VN population
VN_population = sim.Population(
    num_VN_neurons,  # number of neurons
    sim.extra_models.IFCondExpCerebellum(**vn_neuron_params),  # Neuron model
    label="VN",
    additional_parameters={"rb_left_shifts": rbls['vn']})
all_populations["vn"] = VN_population

# Create GrC population
GC_population = sim.Population(
    num_GC_neurons, sim.IF_curr_exp(), label='GrC',
    additional_parameters={"rb_left_shifts": rbls['granule']})
all_populations["granule"] = GC_population

# Create CF population - fake input population that will be substituted by
# external input from robot
CF_population = sim.Population(num_CF_neurons,  # number of sources
                               sim.SpikeSourcePoisson,  # source type
                               {'rate': 0},  # source spike times
                               label="CF",
                               additional_parameters={'seed': 24534}
                               )
all_populations["climbing_fibres"] = CF_population

###############################################################################
# ============================ Create connections =========================== #
###############################################################################

# Create MF-GO connections
mf_go_connections = sim.Projection(MF_population,
                                   GOC_population,
                                   sim.OneToOneConnector(),
                                   sim.StaticSynapse(delay=delay_distr,
                                                     weight=mf_go_weights),
                                   receptor_type='excitatory',
                                   label="mf_goc")
all_projections["mf_goc"] = mf_go_connections

# Create MF-GC and GO-GC connections
float_num_MF_neurons = float(num_MF_neurons)

list_GOC_GC = []
list_MF_GC = []
list_GOC_GC_2 = []
# proj to subpops https://github.com/SpiNNakerManchester/sPyNNaker8/issues/168)
for i in range(num_MF_neurons):
    GC_medium_index = int(round((i / float_num_MF_neurons) * num_GC_neurons))
    GC_lower_index = GC_medium_index - 40
    GC_upper_index = GC_medium_index + 60

    if (GC_lower_index < 0):
        GC_lower_index = 0

    elif (GC_upper_index > num_GC_neurons):
        GC_upper_index = num_GC_neurons

    for j in range(GC_medium_index - GC_lower_index):
        list_GOC_GC.append(
            (i, GC_lower_index + j,
             weight_distr_GO.next(), delay_distr.next())
        )

    for j in range(GC_medium_index + 20 - GC_medium_index):
        list_MF_GC.append(
            (i, GC_medium_index + j,
             weight_distr_MF.next(), delay_distr.next())
        )

    for j in range(GC_upper_index - GC_medium_index - 20):
        list_GOC_GC_2.append(
            (i, GC_medium_index + 20 + j,
             weight_distr_GO.next(), delay_distr.next())
        )

GO_GC_con1 = sim.Projection(GOC_population,
                            GC_population,
                            sim.FromListConnector(list_GOC_GC),
                            label='goc_grc_1',
                            receptor_type='inhibitory')
all_projections["goc_grc_1"] = GO_GC_con1

MF_GC_con2 = sim.Projection(MF_population,
                            GC_population,
                            sim.FromListConnector(list_MF_GC),
                            label='mf_grc',
                            receptor_type='excitatory')

all_projections["mf_grc"] = MF_GC_con2

GO_GC_con3 = sim.Projection(GOC_population,
                            GC_population,
                            sim.FromListConnector(list_GOC_GC_2),
                            label='goc_grc_2',
                            receptor_type='inhibitory')
all_projections["goc_grc_2"] = GO_GC_con3

# Create PC-VN connections
ff_conn_vn = ff_1_to_1_odd_even_mapping_reversed(num_VN_neurons)
assert ff_conn_vn.shape[1] == 2
ff_pc_vn_connections = sim.Projection(
    PC_population, VN_population,
    # sim.OneToOneConnector(),
    sim.FromListConnector(conn_list=ff_conn_vn),
    label='pc_vn',
    synapse_type=sim.StaticSynapse(delay=delay_distr, weight=pc_vn_weights),
    receptor_type='inhibitory')
all_projections["pc_vn"] = ff_pc_vn_connections

# Create MF-VN learning rule - cos

mfvn_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependenceMFVN(
        beta=mfvn_beta, sigma=mfvn_sigma, alpha=mfvn_ltd_constant),
    weight_dependence=sim.extra_models.WeightDependenceMFVN(
        w_min=mfvn_min_weight, w_max=mfvn_max_weight,
        pot_alpha=mfvn_ltp_constant),
    weight=mfvn_initial_weight, delay=delay_distr)

# Create MF to VN connections
mf_vn_connections = sim.Projection(
    MF_population, VN_population, sim.AllToAllConnector(),
    # Needs mapping as FromListConnector to make efficient
    synapse_type=mfvn_plas,
    label='mf_vn',
    receptor_type="excitatory")
all_projections["mf_vn"] = mf_vn_connections

# Create projection from PC to VN -- replaces "TEACHING SIGNAL"
teaching_pc_vn_connections = sim.Projection(
    PC_population, VN_population,
    # sim.OneToOneConnector(),
    sim.FromListConnector(conn_list=ff_conn_vn),
    sim.StaticSynapse(weight=0.0, delay=1.0),
    label='pc_vn_teaching',
    receptor_type="excitatory")  # "TEACHING SIGNAL"
all_projections["pc_vn_teaching"] = teaching_pc_vn_connections

# create PF-PC learning rule - sin
pfpc_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependencePFPC(
        t_peak=pfpc_t_peak, alpha=pfpc_ltd_constant),
    weight_dependence=sim.extra_models.WeightDependencePFPC(
        w_min=pfpc_min_weight, w_max=pfpc_max_weight,
        pot_alpha=pfpc_ltp_constant),
    weight=pfpc_initial_weight, delay=delay_distr)

# Create PF-PC connections
pf_pc_connections = sim.Projection(
    GC_population, PC_population, sim.AllToAllConnector(),
    synapse_type=pfpc_plas,
    label='pf_pc',
    receptor_type="excitatory")
all_projections["pf_pc"] = pf_pc_connections

# Create IO-PC connections. This synapse with "receptor_type=COMPLEX_SPIKE"
# propagates the learning signals that drive the plasticity mechanisms in
# GC-PC synapses
cf_pc_connections = sim.Projection(CF_population,
                                   PC_population,
                                   sim.OneToOneConnector(),
                                   label='cf_pc',
                                   # receptor_type='COMPLEX_SPIKE',
                                   synapse_type=sim.StaticSynapse(
                                       delay=1.0, weight=cf_pc_weights),
                                   receptor_type='excitatory')
all_projections["cf_pc"] = cf_pc_connections

# Instantiate env
head_pos, head_vel = generate_head_position_and_velocity(
    1, slowdown=args.slowdown_factor)
midway_point = 500 * args.slowdown_factor

# perfect eye positions and velocities are exactly out of phase with head
perfect_eye_pos = np.concatenate(
    (head_pos[midway_point:], head_pos[:midway_point]))
perfect_eye_vel = np.concatenate(
    (head_vel[midway_point:], head_vel[:midway_point]))

if USE_MOTION_TARGET:
    head_pos[:midway_point] = -1.0
    head_pos[midway_point:] = 1.0
    head_vel[:] = np.clip(
        np.diff(np.concatenate((head_pos, [head_pos[0]]))) / POS_TO_VEL,
        a_min=-1,
        a_max=1
    )
    perfect_eye_pos = head_pos
    perfect_eye_vel = head_vel

icub_vor_env_model = gym.ICubVorEnv(
    head_pos=head_pos, head_vel=head_vel, perfect_eye_pos=perfect_eye_pos,
    perfect_eye_vel=perfect_eye_vel, error_window_size=sample_time,
    low_error_rate=args.f_base, high_error_rate=args.f_peak,
    wta_decision=args.wta_decision, output_size=num_CF_neurons, gain=gain,
    pos_to_vel=vel_to_pos)
icub_vor_env_pop = sim.Population(ICUB_VOR_VENV_POP_SIZE, icub_vor_env_model)

# Input -> ICubVorEnv projection
# vn_to_icub = sim.Projection(
#     VN_population, icub_vor_env_pop, sim.AllToAllConnector(),
#     label="VN-iCub")
sim.external_devices.activate_live_output_to(VN_population, icub_vor_env_pop)

# ICubVorEnv -> output, setup live output to the SSP vertex
sim.external_devices.activate_live_output_to(
    icub_vor_env_pop, CF_population, "CONTROL")


# ============================  Set up recordings ============================

# Enable relevant recordings
enable_recordings_for(all_populations, full_recordings=args.full_recordings)

# provenance gathering
x = args.filename or "cerebellum_experiment"

structured_provenance_filename = os.path.join(
    result_dir,
    "{}_{}_structured_provenance.npz".format(x, suffix))

if os.path.exists(structured_provenance_filename):
    os.remove(structured_provenance_filename)

# ============================  Set up constraints ============================

# Hard-coding population placements for testing
# MF_population.set_constraint(RadialPlacementFromChipConstraint(0, 0))
# GC_population.set_constraint(RadialPlacementFromChipConstraint(2, 1))
# CF_population.set_constraint(RadialPlacementFromChipConstraint(5, 5))
# VN_population.set_constraint(RadialPlacementFromChipConstraint(4, 1))
# PC_population.set_constraint(RadialPlacementFromChipConstraint(2, 4))
# GOC_population.set_constraint(RadialPlacementFromChipConstraint(1, 2))
# icub_vor_env_pop.set_constraint(RadialPlacementFromChipConstraint(5, 3))

for pop_name, constraint in per_pop_neurons_per_core_constraint.items():
    print("Setting NPC=", constraint, "for", pop_name)
    all_populations[pop_name].set_max_atoms_per_core(constraint)

# ============================  Run simulation ============================

total_runtime = 0
VN_transfer_func = []

print("=" * 80)
print("Running simulation for", runtime)
all_spikes_first_trial = {}
# Record simulation start time (wall clock)
current_error = None
sim_start_time = datetime.now()
# Run the simulation
final_connectivity = {k: [] for k in all_projections.keys()}
icub_snapshots = []
try:
    for run in range(runtime // single_runtime):
        sim.run(single_runtime)
        # Retrieve final network connectivity
        take_connectivity_snapshot(all_projections, final_connectivity)
except Exception as e:  # pylint: disable=broad-except
    print("An exception occurred during execution!")
    traceback.print_exc()
    current_error = e

end_time = datetime.now()
total_time = end_time - start_time
sim_total_time = end_time - sim_start_time

# =====================  Retrieving data from simulation ======================

MF_spikes = MF_population.get_data('spikes')
CF_spikes = CF_population.get_data('spikes')
GC_spikes = GC_population.get_data('all')
GOC_spikes = GOC_population.get_data('spikes')
VN_spikes = VN_population.get_data('all')  # VN_population.get_data('spikes')
PC_spikes = PC_population.get_data('spikes')

mfvn_weights = mf_vn_connections.get('weight', 'list', with_address=False)
pfpc_weights = pf_pc_connections.get('weight', 'list', with_address=False)

# Retrieve recordings for LIF populations
all_spikes = retrieve_all_spikes(all_populations)
other_recordings, neo_all_recordings = \
    retrieve_all_other_recordings(all_populations, args.full_recordings)
# Get the data from the ICubVorEnv pop
results = retrieve_and_package_results(icub_vor_env_pop)
icub_snapshots.append(results)

sim_name = sim.name

sim.end()
print("job done")
# Report time taken
print("Total time elapsed -- " + str(total_time))
# ============================  Plotting some stuff ===========================
# ============================  PAB ANALYSIS ============================

# Compute plot order
plot_order = get_plot_order(all_spikes.keys())
n_plots = float(len(plot_order))

# Check if using neo blocks
neo_all_spikes = {}
for pop, potential_neo_block in all_spikes.items():
    if isinstance(potential_neo_block, neo.Block):
        # make a copy of the spikes dict
        neo_all_spikes[pop] = potential_neo_block
        all_spikes[pop] = convert_spikes(potential_neo_block)

remapped_vn_spikes = remap_odd_even(all_spikes['vn'], all_neurons['vn'])
remapped_cf_spikes = remap_second_half_descending(
    all_spikes['climbing_fibres'], all_neurons['climbing_fibres'])

simulation_parameters = {
    "argparser": vars(args),
    'runtime': runtime,
    'error_window_size': sample_time,
    'vn_spikes': remapped_vn_spikes,
    'cf_spikes': remapped_cf_spikes,
    'perfect_eye_pos': perfect_eye_pos,
    'perfect_eye_vel': perfect_eye_vel,
    'vn_size': all_neurons['vn'],
    'cf_size': all_neurons['climbing_fibres'],
    'gain': gain,
    "git_hash": retrieve_git_commit(),
    "run_end_time": end_time.strftime("%H:%M:%S_%d/%m/%Y"),
    "wall_clock_script_run_time": str(total_time),
    "wall_clock_sim_run_time": str(sim_total_time),
    "n_neurons_per_core": global_n_neurons_per_core,
    "ss_neurons_per_core": ss_neurons_per_core,
    "rbls": rbls,
}

# Save results
if args.suffix:
    suffix = "_" + args.suffix
else:
    suffix = end_time.strftime("_%H%M%S_%d%m%Y")
if args.filename:
    filename = args.filename + str(suffix)
else:
    filename = "full_cerebellum_test" + str(suffix)

if current_error:
    filename = "error_" + filename

# this would be the best point to look at the database
save_provenance_to_file_from_database(
    structured_provenance_filename, sim_name)

# Try to read the structured provenance
try:
    struct_prov = np.load(structured_provenance_filename, allow_pickle=True)
except Exception:  # pylint: disable=broad-except
    struct_prov = {}
    print("Failed to retrieve structured provenance")
    traceback.print_exc()

to_save_struct_prov = {k: v for k, v in struct_prov.items()}

# Save results to file in [by default] the `results/' directory
results_file = os.path.join(result_dir, filename)
np.savez_compressed(
    results_file, simulation_parameters=simulation_parameters,
    all_spikes=all_spikes, neo_all_spikes=neo_all_spikes,
    other_recordings=other_recordings,
    all_neurons=all_neurons, all_neuron_params=all_neuron_params,
    final_connectivity=final_connectivity,
    initial_connectivity=initial_connectivity,
    simtime=runtime, conn_params=CONNECTIVITY_MAP, cell_params=neuron_params,
    per_pop_neurons_per_core_constraint=per_pop_neurons_per_core_constraint,
    icub_snapshots=icub_snapshots,
    structured_provenance=to_save_struct_prov,
    structured_provenance_filename=structured_provenance_filename)

# Report time taken
print("Results stored in  -- " + filename)

# Report time taken
print("Total time elapsed -- " + str(total_time))
analyse_run(results_file=results_file,
            filename=filename,
            suffix=suffix)

if not args.no_provenance:
    provenance_analysis(
        structured_provenance_filename,
        fig_folder=fig_folder + filename + "/provenance_figures")

# Report time taken
print("Results stored in  -- " + filename)

# Report time taken
print("Total time elapsed -- " + str(total_time))
