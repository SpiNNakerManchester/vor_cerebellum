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
This script tests  the operation of the windowing procedure for
PF + CF on the PF-PC weights, i.e. the weights from GrC to PC.

Check provenance to verify the correct number of spikes being counted.
"""
import pyNN.spiNNaker as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from brian2.units import nS, uS
from vor_cerebellum.parameters import (pfpc_min_weight, pfpc_max_weight,
                                       pfpc_ltp_constant, pfpc_t_peak,
                                       rbls, neuron_params)

runtime = 500
initial_weight = 0.05 * nS / uS
plastic_delay = 4

p.setup(1)  # simulation timestep (ms)
purkinje_cell = p.Population(1,  # number of neurons
                             p.extra_models.IFCondExpCerebellum(**neuron_params),  # Neuron model
                             label="PC",
                             additional_parameters={"rb_left_shifts": rbls['purkinje']}
                             )

# Spike source to send spike via synapse
pf_spike_times = [50, 60, 65, 85, 101, 400]

granular_cell = p.Population(1,  # number of sources
                             p.SpikeSourceArray,  # source type
                             {'spike_times': pf_spike_times},  # source spike times
                             label="GrC"  # identifier
                             )

# Spike source to send spike via synapse from climbing fibre
# cf_spike_times = [55, 80, 90, 95, 96, 201]  # , 104, 107, 246]
# Modifying spike times to be within 1 time step of pf_spike_times
cf_spike_times = [50, 80, 89, 95, 97, 201]
climbing_fibre = p.Population(1,  # number of sources
                              p.SpikeSourceArray,  # source type
                              {'spike_times': cf_spike_times},  # source spike times
                              label="CF"  # identifier
                              )

# Create projection from GC to PC
pfpc_plas = p.STDPMechanism(
    timing_dependence=p.extra_models.TimingDependencePFPC(t_peak=pfpc_t_peak),
    weight_dependence=p.extra_models.WeightDependencePFPC(w_min=pfpc_min_weight,
                                                          w_max=pfpc_max_weight,
                                                          pot_alpha=pfpc_ltp_constant),
    weight=initial_weight, delay=plastic_delay)

synapse_pfpc = p.Projection(
    granular_cell, purkinje_cell, p.AllToAllConnector(),
    synapse_type=pfpc_plas, receptor_type="excitatory")

# Create projection from CF to PC
synapse = p.Projection(
    climbing_fibre, purkinje_cell, p.OneToOneConnector(),
    p.StaticSynapse(weight=0.0, delay=1), receptor_type="excitatory")

granular_cell.record('spikes')
climbing_fibre.record('spikes')
purkinje_cell.record("all")

p.run(runtime)

granluar_cell_spikes = granular_cell.get_data('spikes')
climbing_fibre_spikes = climbing_fibre.get_data('spikes')
purkinje_data = purkinje_cell.get_data()

pf_weights = synapse_pfpc.get('weight', 'list', with_address=False)
p.end()

print(pf_weights)

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

plt.savefig("figures/pc_windowing.png", dpi=400)

print("Job Complete")
