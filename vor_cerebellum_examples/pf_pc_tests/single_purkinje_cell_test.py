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

from __future__ import print_function
import os
import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from vor_cerebellum.parameters import (pfpc_min_weight, pfpc_max_weight,
                                       pfpc_ltp_constant, pfpc_t_peak,
                                       pfpc_plasticity_delay,
                                       rbls, neuron_params)
from vor_cerebellum.provenance_analysis import (
    provenance_analysis, save_provenance_to_file_from_database)


sim.setup(1, min_delay=1, max_delay=15)  # simulation timestep (ms)
runtime = 500
# Learning parameters
# min_weight = 0
# max_weight = 0.1
# pot_alpha = 0.01
# t_peak = 100
initial_weight = 0.05
# plastic_delay = 4

purkinje_cell = sim.Population(1,  # number of neurons
                               sim.extra_models.IFCondExpCerebellum(
                                   **neuron_params),  # Neuron model
                               additional_parameters={
                                   "rb_left_shifts": rbls['purkinje']},
                               label="PC"  # identifier
                               )

# Spike source to send spike via synapse
spike_times = [50, 150, 270]

granular_cell = sim.Population(1,  # number of sources
                               sim.SpikeSourceArray,  # source type
                               {'spike_times': spike_times},  # spike times
                               label="GrC"  # identifier
                               )

# Spike source to send spike via synapse from climbing fibre
spike_times_2 = [100, 104, 107, 246]
climbing_fibre = sim.Population(1,  # number of sources
                                sim.SpikeSourceArray,  # source type
                                {'spike_times': spike_times_2},  # spike times
                                label="CF"  # identifier
                                )

# Create projection from GC to PC
pfpc_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependencePFPC(
        t_peak=pfpc_t_peak),
    weight_dependence=sim.extra_models.WeightDependencePFPC(
        w_min=pfpc_min_weight, w_max=pfpc_max_weight,
        pot_alpha=pfpc_ltp_constant),
    weight=initial_weight, delay=pfpc_plasticity_delay)

synapse_pfpc = sim.Projection(
    granular_cell, purkinje_cell, sim.AllToAllConnector(),
    synapse_type=pfpc_plas, receptor_type="excitatory")

# Create projection from CF to PC
synapse = sim.Projection(
    climbing_fibre, purkinje_cell, sim.OneToOneConnector(),
    sim.StaticSynapse(weight=0.0, delay=1), receptor_type="excitatory")

granular_cell.record('spikes')
climbing_fibre.record('spikes')
purkinje_cell.record("all")

sim.run(runtime)

granluar_cell_spikes = granular_cell.get_data('spikes')
climbing_fibre_spikes = climbing_fibre.get_data('spikes')
purkinje_data = purkinje_cell.get_data()

pf_weights = synapse_pfpc.get('weight', 'list', with_address=False)

structured_provenance_filename = "single_pc_structured_provenance.npz"
if os.path.exists(structured_provenance_filename):
    os.remove(structured_provenance_filename)

sim_name = sim.name
# TODO: it may be possible to take the simulator out of this call
save_provenance_to_file_from_database(
    structured_provenance_filename, sim_name)

sim.end()
print("Final PF-PC weight", pf_weights)

# Plot
F = Figure(
    # plot data for postsynaptic neuron
    Panel(granluar_cell_spikes.segments[0].spiketrains,
          ylabel=granular_cell.label,
          yticks=True, markersize=2, xlim=(0, runtime)),

    Panel(climbing_fibre_spikes.segments[0].spiketrains,
          ylabel=climbing_fibre.label,
          yticks=True, markersize=2, xlim=(0, runtime)),

    Panel(purkinje_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[purkinje_cell.label], yticks=True, xlim=(0, runtime)),

    Panel(purkinje_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[purkinje_cell.label], yticks=True, xlim=(0, runtime)),

    Panel(purkinje_data.segments[0].spiketrains,
          ylabel=purkinje_cell.label,
          yticks=True, markersize=2, xlim=(0, runtime)),
)

plt.savefig("figures/single_pc_test.png", dpi=400)

provenance_analysis(structured_provenance_filename, "provenance_figures/")

print("Done")
