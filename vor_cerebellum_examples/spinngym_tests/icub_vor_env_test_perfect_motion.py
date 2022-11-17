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

import pyNN.spiNNaker as p
import numpy as np
import spinn_gym as gym
from vor_cerebellum.utilities import (
    generate_head_position_and_velocity, remap_odd_even,
    remap_second_half_descending, retrieve_and_package_results, plot_results,
    ICUB_VOR_VENV_POP_SIZE)

# Parameter definition
runtime = 3000
# Build input SSP and output population
input_size = 200  # neurons
output_size = 200  # neurons
gain = 20.0

slowdown_factor = 2
midway_point = 500 * slowdown_factor
vel_to_pos = 1 / (2 * np.pi * slowdown_factor * gain)
pos_to_vel = 1 / (0.001 * 2 * np.pi * slowdown_factor)
error_window_size = 10  # ms

head_pos, head_vel = generate_head_position_and_velocity(
    1, slowdown=slowdown_factor)

# perfect eye positions and velocities are exactly out of phase with head
perfect_eye_pos = np.concatenate(
    (head_pos[midway_point:], head_pos[:midway_point]))
perfect_eye_vel = np.concatenate(
    (head_vel[midway_point:], head_vel[:midway_point]))

input_spike_times = [[] for _ in range(input_size)]
# the constant number (0.000031) is the effect of a single spike on the head
# position
# assert (np.isclose(np.abs(
#     np.diff(head_pos)[0]), no_required_spikes_per_chunk * 0.000031), 0.001)
sub_head_pos = np.diff(head_vel)
head_movement_per_spike = 2 ** (-15) * gain
sub_eye_pos = np.diff(np.concatenate((perfect_eye_pos, [perfect_eye_pos[0]])))

# no_required_spikes_per_chunk = 200
no_required_spikes_per_chunk = np.ceil(
    np.abs(sub_head_pos[0]) / head_movement_per_spike)

# build ICubVorEnv model
adjusted_window = 1000 * slowdown_factor
npc_limit = 200  # 25
no_input_cores = int(input_size / npc_limit)
input_spike_times = [[] for _ in range(input_size)]
for ts in range(runtime - 1):
    # if 1000 <= ts < 2000:
    #     continue
    sgn = np.sign(sub_eye_pos[ts % adjusted_window])
    spikes_during_chunk = np.ceil(
        np.abs(sub_eye_pos[ts % adjusted_window]) / head_movement_per_spike)
    for i in range(int(spikes_during_chunk)):
        x = int(sgn <= 0)
        input_spike_times[(i % no_input_cores) * npc_limit + x].append(ts)

# Setup
p.setup(timestep=1.0)
p.set_number_of_neurons_per_core(p.SpikeSourcePoisson, 50)
p.set_number_of_neurons_per_core(p.SpikeSourceArray, npc_limit)
input_pop = p.Population(
    input_size, p.SpikeSourceArray(spike_times=input_spike_times))

output_pop = p.Population(output_size, p.SpikeSourcePoisson(rate=0))

# Instantiate venv
icub_vor_env_model = gym.ICubVorEnv(
    head_pos=head_pos, head_vel=head_vel,
    perfect_eye_pos=perfect_eye_pos, perfect_eye_vel=perfect_eye_vel,
    error_window_size=error_window_size, wta_decision=False,
    low_error_rate=2,
    high_error_rate=20,
    output_size=output_size,
    gain=gain,
    pos_to_vel=vel_to_pos
)
icub_vor_env_pop = p.Population(ICUB_VOR_VENV_POP_SIZE, icub_vor_env_model)

# Set recording for input and output pop (env pop records by default)
input_pop.record('spikes')
output_pop.record('spikes')

# Input -> ICubVorEnv projection
# i2a = p.Projection(input_pop, icub_vor_env_pop, p.AllToAllConnector())
p.external_devices.activate_live_output_to(input_pop, icub_vor_env_pop)

# ICubVorEnv -> output, setup live output to the SSP vertex
p.external_devices.activate_live_output_to(
    icub_vor_env_pop, output_pop, "CONTROL")

# Run the simulation
p.run(runtime)

# Get the data from the ICubVorEnv pop
results = retrieve_and_package_results(icub_vor_env_pop)

# get the spike data from input and output
spikes_in_spin = input_pop.spinnaker_get_data('spikes')
spikes_out_spin = output_pop.spinnaker_get_data('spikes')

# end simulation
p.end()

remapped_vn_spikes = remap_odd_even(spikes_in_spin, input_size)
remapped_cf_spikes = remap_second_half_descending(spikes_out_spin, output_size)

simulation_parameters = {
    'runtime': runtime,
    'error_window_size': error_window_size,
    'vn_spikes': remapped_vn_spikes,
    'cf_spikes': remapped_cf_spikes,
    'perfect_eye_pos': perfect_eye_pos,
    'perfect_eye_vel': perfect_eye_vel,
    'vn_size': input_size,
    'cf_size': output_size,
    'gain': gain
}

# plot the data from the ICubVorEnv pop
plot_results(results_dict=results, simulation_parameters=simulation_parameters,
             all_spikes={
                 'purkinje': np.empty([0, 2])
             },
             name="figures/spinngym_icub_vor_test_perfect")

print("Done")
