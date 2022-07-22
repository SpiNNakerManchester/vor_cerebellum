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
import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from vor_cerebellum.parameters import (mfvn_min_weight, mfvn_max_weight,
                                       mfvn_initial_weight,
                                       mfvn_ltp_constant,
                                       mfvn_beta, mfvn_sigma,
                                       mfvn_plasticity_delay,
                                       rbls, neuron_params)

sim.setup(1)  # simulation timestep (ms)
runtime = 500

vestibular_nuclei = sim.Population(1,  # number of neurons
                                   sim.extra_models.IFCondExpCerebellum(**neuron_params),  # Neuron model
                                   label="Vestibular Nuclei",
                                   additional_parameters={"rb_left_shifts": rbls['vn']}
                                   )

# Spike source to send spike via synapse
mf_spike_times = [50, 60, 65]  # , 150, 175, 180, 190, 240, 250, 255,
#                270, 300, 345, 350, 360, 370, 400, 422, 425, 427, 429]

mossy_fibre_src = sim.Population(1,  # number of sources
                                 sim.SpikeSourceArray,  # source type
                                 {'spike_times': mf_spike_times},  # source spike times
                                 label="src1"  # identifier
                                 )

# Spike source to send spike via synapse from climbing fibre
pc_spike_times = [60]  # , 104, 107, 246]
purkinje_cell_src = sim.Population(1,  # number of sources
                                   sim.SpikeSourceArray,  # source type
                                   {'spike_times': pc_spike_times},  # source spike times
                                   label="purkinje_cell_src"  # identifier
                                   )

# Create projection from GC to PC
mfvn_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependenceMFVN(beta=mfvn_beta,
                                                            sigma=mfvn_sigma),
    weight_dependence=sim.extra_models.WeightDependenceMFVN(w_min=mfvn_min_weight,
                                                            w_max=mfvn_max_weight,
                                                            pot_alpha=mfvn_ltp_constant),
    weight=mfvn_initial_weight, delay=mfvn_plasticity_delay)

synapse_mfvn = sim.Projection(
    mossy_fibre_src, vestibular_nuclei, sim.AllToAllConnector(),
    synapse_type=mfvn_plas, receptor_type="excitatory")

# Create projection from PC to VN
synapse = sim.Projection(
    purkinje_cell_src, vestibular_nuclei, sim.OneToOneConnector(),
    sim.StaticSynapse(weight=0.0, delay=1), receptor_type="excitatory")

mossy_fibre_src.record('spikes')
purkinje_cell_src.record('spikes')
vestibular_nuclei.record("all")

sim.run(runtime)

mossy_fibre_src_spikes = mossy_fibre_src.get_data('spikes')
purkinje_cell_src_spikes = purkinje_cell_src.get_data('spikes')
vestibular_nuclei_data = vestibular_nuclei.get_data()

mf_weights = synapse_mfvn.get('weight', 'list', with_address=False)

sim.end()

print("\n {}".format(mf_weights))

# Plot
F = Figure(
    # plot data for postsynaptic neuron
    Panel(mossy_fibre_src_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, runtime)),

    Panel(purkinje_cell_src_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, runtime)),

    Panel(vestibular_nuclei_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[vestibular_nuclei.label], yticks=True, xlim=(0, runtime)),

    Panel(vestibular_nuclei_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[vestibular_nuclei.label], yticks=True, xlim=(0, runtime)),

    Panel(vestibular_nuclei_data.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, runtime)),
)

plt.savefig("figures/vn_mf_3_spike_windowing_test.png", dpi=400)
print("Job Complete")
