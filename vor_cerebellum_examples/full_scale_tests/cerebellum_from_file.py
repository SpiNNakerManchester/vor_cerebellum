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

# PAB imports
import traceback
import neo
import numpy as np
import os
from datetime import datetime
# general parameters
from vor_cerebellum.parameters import (CONNECTIVITY_MAP, rbls, neuron_params)
from vor_cerebellum.utilities import (
    write_header, write_value, ff_1_to_1_odd_even_mapping_reversed,
    generate_head_position_and_velocity, POS_TO_VEL, ICUB_VOR_VENV_POP_SIZE,
    enable_recordings_for, retrieve_all_spikes, retrieve_all_other_recordings,
    retrieve_and_package_results, get_plot_order, convert_spikes,
    remap_odd_even, remap_second_half_descending, retrieve_git_commit,
    result_dir, analyse_run, sensorial_activity)
# Imports for SpiNNGym env
import spinn_gym as gym
from vor_cerebellum.vor_argparser import args

# Record SCRIPT start time (wall clock)
start_time = datetime.now()

write_header("Re-building a pre-trained network from: " + args.path_to_input)

# Get npz archive
previous_run_data = np.load(args.path_to_input, allow_pickle=True)

# Get required parameters out of the
all_spikes = previous_run_data['all_spikes'].ravel()[0]
all_neurons = previous_run_data['all_neurons'].ravel()[0]
# TODO make sure all new runs save this info
# all_neuron_params = previous_run_data['all_neuron_params'].ravel()[0]
simulation_parameters = previous_run_data['simulation_parameters'].ravel()[0]
other_recordings = previous_run_data['other_recordings'].ravel()[0]
final_connectivity = previous_run_data['final_connectivity'].ravel()[0]
simtime = previous_run_data['simtime']
cell_params = previous_run_data['cell_params'].ravel()[0]
per_pop_neurons_per_core_constraint = previous_run_data[
    'per_pop_neurons_per_core_constraint'].ravel()[0]
icub_snapshots = previous_run_data['icub_snapshots']

# Which snapshot to use?
if args.snapshot.lower() == "last":
    write_header("Rebuilding network using the LAST extracted weights")
    snap_index = -1
elif args.snapshot.lower() == "best":
    write_header("Rebuilding network using the BEST extracted weights")
    # unpacking results
    errors = icub_snapshots[-1]['errors']

    # unpacking simulation params
    runtime = simulation_parameters['runtime']
    error_window_size = simulation_parameters['error_window_size']
    perfect_eye_pos = simulation_parameters['perfect_eye_pos']
    vn_size = simulation_parameters['vn_size']
    cf_size = simulation_parameters['cf_size']

    pattern_period = perfect_eye_pos.shape[0]  # in ms

    n_plots = runtime / (pattern_period)
    error_windows_per_pattern = int(pattern_period / error_window_size)
    reshaped_error = errors.reshape(
        errors.size // error_windows_per_pattern, error_windows_per_pattern)
    maes = np.mean(np.abs(reshaped_error), axis=1)
    snap_index = np.argmin(maes)

write_value("Snapshot index", snap_index)
# Network parameters
num_MF_neurons = all_neurons['mossy_fibres']
num_GC_neurons = all_neurons['granule']
num_GOC_neurons = all_neurons['golgi']
num_PC_neurons = all_neurons['purkinje']
num_VN_neurons = all_neurons['vn']
num_CF_neurons = all_neurons['climbing_fibres']

# Starting to record additional parameters

USE_MOTION_TARGET = args.target_reaching  # default false

# cerebellum test bench
runtime = args.simtime  # default=10k ms
suffix = args.suffix

single_runtime = args.single_simtime  # default=10k ms
sample_time = args.error_window_size  # default 10 ms

# SpiNNGym settings
gain = args.gain

all_populations = {

}

initial_connectivity = {

}

all_projections = {

}

neo_all_recordings = {}

global_n_neurons_per_core = 50
ss_neurons_per_core = 25
pressured_npc = 10
per_pop_neurons_per_core_constraint = {
    'mossy_fibres': global_n_neurons_per_core,
    'granule': global_n_neurons_per_core,
    'golgi': global_n_neurons_per_core,
    'purkinje': pressured_npc,
    'vn': pressured_npc,
    'climbing_fibres': pressured_npc,
}

sim.setup(timestep=1., min_delay=1, max_delay=15)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, ss_neurons_per_core)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, ss_neurons_per_core)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp, global_n_neurons_per_core)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 128)
sim.set_number_of_neurons_per_core(
    sim.extra_models.IFCondExpCerebellum, global_n_neurons_per_core)
sim.set_number_of_neurons_per_core(
    sim.extra_models.SpikeSourcePoissonVariable, ss_neurons_per_core)

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
    desired_pos = -np.sin(t * 2 * np.pi) * MAX_AMPLITUDE * RELATIVE_AMPLITUDE
    _head_pos.append(desired_pos)
    _head_vel.append(desired_speed)

normalised_head_pos = (np.asarray(_head_pos) + 0.8) / 1.6
normalised_head_vel = (
    np.asarray(_head_vel) + 0.8 * 2 * np.pi) / (1.6 * 2 * np.pi)

##############################################################################
# ============================ Create populations ========================== #
##############################################################################

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
GoC_population = sim.Population(
    num_GOC_neurons, sim.IF_cond_exp(**cell_params), label='GoC',
    additional_parameters={"rb_left_shifts": rbls['golgi']})
all_populations["golgi"] = GoC_population

# create PC population
PC_population = sim.Population(num_PC_neurons,  # number of neurons
                               sim.IF_cond_exp(**cell_params),  # Neuron model
                               label="PC",
                               additional_parameters={
                                   "rb_left_shifts": rbls['purkinje']}
                               )
all_populations["purkinje"] = PC_population

# create VN population
VN_population = sim.Population(num_VN_neurons,  # number of neurons
                               sim.IF_cond_exp(**cell_params),  # Neuron model
                               label="VN",
                               additional_parameters={
                                   "rb_left_shifts": rbls['vn']}
                               )
all_populations["vn"] = VN_population

# Create GrC population
GrC_population = sim.Population(num_GC_neurons, sim.IF_curr_exp(), label='GrC',
                                additional_parameters={
                                    "rb_left_shifts": rbls['granule']}
                                )
all_populations["granule"] = GrC_population

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
mf_go_connections = sim.Projection(
    MF_population, GoC_population,
    sim.FromListConnector(final_connectivity['mf_goc'][snap_index]),
    receptor_type='excitatory', label="mf_goc")
all_projections["mf_goc"] = mf_go_connections

# Create MF-GC and GO-GC connections
float_num_MF_neurons = float(num_MF_neurons)

list_GOC_GC = []
list_MF_GC = []
list_GOC_GC_2 = []

GO_GC_con1 = sim.Projection(
    GoC_population, GrC_population,
    sim.FromListConnector(final_connectivity['goc_grc_1'][snap_index]),
    label='goc_grc_1', receptor_type='inhibitory')  # this should be inhibitory
all_projections["goc_grc_1"] = GO_GC_con1

MF_GC_con2 = sim.Projection(
    MF_population, GrC_population,
    sim.FromListConnector(final_connectivity['mf_grc'][snap_index]),
    label='mf_grc', receptor_type='excitatory')

all_projections["mf_grc"] = MF_GC_con2

GO_GC_con3 = sim.Projection(
    GoC_population, GrC_population,
    sim.FromListConnector(final_connectivity['goc_grc_2'][snap_index]),
    label='goc_grc_2', receptor_type='inhibitory')
all_projections["goc_grc_2"] = GO_GC_con3

# Create PC-VN connections
ff_conn_vn = ff_1_to_1_odd_even_mapping_reversed(num_VN_neurons)
assert ff_conn_vn.shape[1] == 2
ff_pc_vn_connections = sim.Projection(
    PC_population, VN_population,
    sim.FromListConnector(conn_list=final_connectivity['pc_vn'][snap_index]),
    label='pc_vn', receptor_type='inhibitory')
all_projections["pc_vn"] = ff_pc_vn_connections

# Create MF to VN connections
mf_vn_connections = sim.Projection(
    MF_population, VN_population,
    sim.FromListConnector(final_connectivity['mf_vn'][snap_index]),
    label='mf_vn', receptor_type="excitatory")
all_projections["mf_vn"] = mf_vn_connections

# cre
# Create PF-PC connections
pf_pc_connections = sim.Projection(
    GrC_population, PC_population,
    sim.FromListConnector(final_connectivity['pf_pc'][snap_index]),
    label='pf_pc', receptor_type="excitatory")
all_projections["pf_pc"] = pf_pc_connections

# Create IO-PC connections. This synapse with "receptor_type=COMPLEX_SPIKE"
# propagates the learning signals that drive the plasticity mechanisms in
# GC-PC synapses
# cf_pc_connections = sim.Projection(
#     CF_population, PC_population, sim.OneToOneConnector(),
#     label='cf_pc',
#     # receptor_type='COMPLEX_SPIKE',
#     synapse_type=sim.StaticSynapse(delay=1.0,
#                                    weight=cf_pc_weights),
#     receptor_type='excitatory')
# all_projections["cf_pc"] = cf_pc_connections

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
    wta_decision=args.wta_decision, output_size=num_CF_neurons, gain=gain)
icub_vor_env_pop = sim.Population(ICUB_VOR_VENV_POP_SIZE, icub_vor_env_model)

# Input -> ICubVorEnv projection
vn_to_icub = sim.Projection(
    VN_population, icub_vor_env_pop, sim.AllToAllConnector(), label="VN-iCub")

# ICubVorEnv -> output, setup live output to the SSP vertex
sim.external_devices.activate_live_output_to(
    icub_vor_env_pop, CF_population, "CONTROL")

# ============================  Set up recordings ============================

# Enable relevant recordings
enable_recordings_for(all_populations, full_recordings=args.full_recordings)

# ============================  Set up constraints ============================

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
new_icub_snapshots = []
try:
    for run in range(runtime // single_runtime):
        sim.run(single_runtime)
except Exception as e:
    print("An exception occurred during execution!")
    traceback.print_exc()
    current_error = e

end_time = datetime.now()
total_time = end_time - start_time
sim_total_time = end_time - sim_start_time

# ==================  Retrieving data from simulation ========================

MF_spikes = MF_population.get_data('spikes')
CF_spikes = CF_population.get_data('spikes')
GC_spikes = GrC_population.get_data('all')
GOC_spikes = GoC_population.get_data('spikes')
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
new_icub_snapshots.append(results)

sim.end()
print("job done")
# Report time taken
print("Total time elapsed -- " + str(total_time))
# ============================  Plotting some stuff ==========================
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
    filename = "network_rebuild_test" + str(suffix)

if current_error:
    filename = "error_" + filename

# Save results to file in [by default] the `results/' directory
results_file = os.path.join(result_dir, filename)
np.savez_compressed(
    results_file, simulation_parameters=simulation_parameters,
    all_spikes=all_spikes, neo_all_spikes=neo_all_spikes,
    other_recordings=other_recordings, all_neurons=all_neurons,
    final_connectivity=final_connectivity,
    initial_connectivity=initial_connectivity,
    simtime=runtime, conn_params=CONNECTIVITY_MAP, cell_params=neuron_params,
    per_pop_neurons_per_core_constraint=per_pop_neurons_per_core_constraint,
    training_icub_snapshots=icub_snapshots, icub_snapshots=new_icub_snapshots,
    testing_icub_snapshots=new_icub_snapshots)

# Report time taken
print("Results stored in  -- " + filename)

# Report time taken
print("Total time elapsed -- " + str(total_time))
analyse_run(results_file=results_file,
            filename=filename,
            suffix=suffix)

# Report time taken
print("Results stored in  -- " + filename)

# Report time taken
print("Total time elapsed -- " + str(total_time))
