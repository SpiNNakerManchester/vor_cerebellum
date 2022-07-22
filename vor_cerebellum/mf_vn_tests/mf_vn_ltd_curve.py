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

"""
This script tests  the operation of the LTD for
MF-VN weights

Check provenance to verify the correct number of spikes being counted.
"""
import pyNN.spiNNaker as sim
from vor_cerebellum.utilities import *
import matplotlib.pyplot as plt
import numpy as np
from vor_cerebellum.parameters import (mfvn_min_weight, mfvn_max_weight,
                                       mfvn_initial_weight,
                                       mfvn_ltp_constant,
                                       mfvn_beta, mfvn_sigma,
                                       mfvn_plasticity_delay,
                                       mfvn_ltd_constant,
                                       rbls, scaling_factor,
                                       vn_neuron_params)

runtime = 300
n_timesteps = 200

all_projections = []
all_populations = []
all_sources = []
final_connectivity = []
final_input_spikes = []
final_recordings = []

sim.setup(1)  # simulation timestep (ms)
pc_spike_times = [n_timesteps]
purkinje_cell = sim.Population(1,  # number of sources
                               sim.SpikeSourceArray,  # source type
                               {'spike_times': pc_spike_times},  # source spike times
                               label="PC"  # identifier
                               )

final_mf_spike_time = n_timesteps + 50

all_mf_spike_times = []

for curr_timestep_diff in range(n_timesteps):
    vn_cell = sim.Population(1,  # number of neurons
                             sim.extra_models.IFCondExpCerebellum(**vn_neuron_params),  # Neuron model
                             label="VN" + str(curr_timestep_diff),
                             additional_parameters={"rb_left_shifts": rbls['vn']}
                             )
    vn_cell.initialize(v=vn_neuron_params['v_rest'])

    # Spike source to send spike via synapse
    mf_spike_times = [curr_timestep_diff, final_mf_spike_time]
    all_mf_spike_times.append(mf_spike_times)

    mf_pop = sim.Population(1,  # number of sources
                            sim.SpikeSourceArray,  # source type
                            {'spike_times': mf_spike_times},  # source spike times
                            label="MF" + str(curr_timestep_diff)  # identifier
                            )
    all_sources.append(mf_pop)

    # Create projection from GC to PC
    mfvn_plas = sim.STDPMechanism(
        timing_dependence=sim.extra_models.TimingDependenceMFVN(beta=mfvn_beta,
                                                                sigma=mfvn_sigma,
                                                                alpha=mfvn_ltd_constant),
        weight_dependence=sim.extra_models.WeightDependenceMFVN(w_min=mfvn_min_weight,
                                                                w_max=mfvn_max_weight,
                                                                pot_alpha=mfvn_ltp_constant),
        weight=mfvn_initial_weight, delay=mfvn_plasticity_delay)

    synapse_mfvn = sim.Projection(
        mf_pop, vn_cell, sim.AllToAllConnector(),
        synapse_type=mfvn_plas, receptor_type="excitatory")
    all_projections.append(synapse_mfvn)

    # Create projection from PC to VN
    pc_vn_synapse = sim.Projection(
        purkinje_cell, vn_cell, sim.OneToOneConnector(),
        sim.StaticSynapse(weight=0.0, delay=1), receptor_type="excitatory")

    mf_pop.record("spikes")
    vn_cell.record("all")
    all_populations.append(vn_cell)

purkinje_cell.record('spikes')
sim.run(runtime)

climbing_fibre_spikes = purkinje_cell.get_data('spikes')

for i in range(n_timesteps):
    final_recordings.append(all_populations[i].get_data())
    final_input_spikes.append(all_sources[i].spinnaker_get_data("spikes"))
    final_connectivity.append(all_projections[i].get('weight', 'list', with_address=False))

pc_vn_synapse_weight = pc_vn_synapse.get('weight', 'list', with_address=False)

connection_strength = np.asarray(final_connectivity).ravel() / scaling_factor
recorded_pc_spikes = np.asarray(final_input_spikes)

sim.end()

all_mf_spike_times = np.asarray(all_mf_spike_times)

assert np.all(all_mf_spike_times[:, 1] == final_mf_spike_time)
assert recorded_pc_spikes.shape[1] == 2
assert np.all(recorded_pc_spikes[:, :, 1][:, 1] == final_mf_spike_time)

write_header("MF-VN LTD Curve")
write_value("PC-VN weight", pc_vn_synapse_weight)
write_value("Initial MF-VN weight", mfvn_initial_weight / scaling_factor)
write_value("MF-VN constant LTP", mfvn_ltp_constant / scaling_factor)
write_value("MF-VN LTD scaling constant", mfvn_ltd_constant)

voltage_matrix = np.ones((n_timesteps, runtime)) * np.nan
packet_matrix = np.ones((n_timesteps, runtime)) * np.nan
gsyn_exc_matrix = np.ones((n_timesteps, runtime)) * np.nan

for i, block in enumerate(final_recordings):
    voltage_matrix[i] = np.array(block.filter(name='v')).ravel() / scaling_factor
    packet_matrix[i] = np.array(block.filter(name='packets-per-timestep')).ravel()
    gsyn_exc_matrix[i] = np.array(block.filter(name='gsyn_exc')).ravel() / scaling_factor

f = plt.figure(1, figsize=(12, 9), dpi=500)
plt.close(f)

for conn_matrix, mat_name, cbar_label in zip(
        [voltage_matrix, packet_matrix, gsyn_exc_matrix],
        ["voltage_matrix", "packet_matrix", "gsyn_exc_matrix"],
        ["Potential (mV)", "# of packets", "$g_{syn}$ (uS)"]):
    f = plt.figure(1, figsize=(12, 9), dpi=500)
    im = plt.imshow(conn_matrix,
                    interpolation='none',
                    extent=[0, runtime,
                            0, n_timesteps],
                    origin='lower')
    ax = plt.gca()
    ax.set_aspect('auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Source Neuron ID")

    save_figure(plt, os.path.join('figures/',
                                  "mf_vn_{}".format(mat_name)),
                extensions=['.png', ])
    plt.close(f)

f = plt.figure(figsize=(12, 9), dpi=500)
plt.plot(connection_strength[::-1])
plt.xlabel("Time difference (ms)")
plt.title("MF-VN LTD curve")
plt.ylabel("Syn. Weight (uS)")
plt.savefig("figures/mf_vn_ltd_curve.png")

print("Job Complete")
