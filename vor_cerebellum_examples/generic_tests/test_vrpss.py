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

from pyNN.utility.plotting import Figure, Panel

import numpy as np
import os
import matplotlib.pyplot as plt

# PAB imports
# general parameters
# from vor_cerebellum.parameters import (L_RATE, H_RATE, rbls, neuron_params)
# # MF-VN params
# from vor_cerebellum.parameters import (mfvn_min_weight, mfvn_max_weight,
#                                        mfvn_initial_weight,
#                                        mfvn_ltp_constant,
#                                        mfvn_beta, mfvn_sigma)
# # PF-PC params
# from vor_cerebellum.parameters import (pfpc_min_weight, pfpc_max_weight,
#                                        pfpc_initial_weight,
#                                        pfpc_ltp_constant, pfpc_t_peak)
from vor_cerebellum.utilities import (
    save_figure, sensorial_activity, convert_spikes, use_display_name,
    fig_folder)

runtime = 1000

samples_in_repeat = 99
sample_time = 10
repeats = 1

num_MF_neurons = 100

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

all_mf_rates = np.ones((num_MF_neurons, runtime // sample_time)) * np.nan
all_mf_starts = np.repeat(
    [np.arange(runtime // sample_time) * sample_time], num_MF_neurons, axis=0)
all_mf_durations = np.ones(
    (num_MF_neurons, runtime // sample_time)) * sample_time
for i in range(runtime // sample_time):
    current_rates = sensorial_activity(_head_pos[i*10], _head_vel[i*10])[0]
    all_mf_rates[:, i] = current_rates


sim.setup(1)
sim.set_number_of_neurons_per_core(
    sim.extra_models.SpikeSourcePoissonVariable, 25)
MF_population = sim.Population(num_MF_neurons,  # number of sources
                               sim.extra_models.SpikeSourcePoissonVariable,
                               {'rates': all_mf_rates,
                                'starts': all_mf_starts,
                                'durations': all_mf_durations
                                },  # source spike times
                               label="MF",
                               additional_parameters={'seed': 24534}
                               )
MF_population.record(['spikes'])

sim.run(runtime)

MF_spikes = MF_population.get_data('spikes')

sim.end()
# Retrieve recordings

F = Figure(
    Panel(MF_spikes.segments[0].spiketrains,
          yticks=True, markersize=2,
          xlabel='MF_spikes'),
)
save_figure(plt, os.path.join(fig_folder, "mf_spikes"),
            extensions=['.png', ])

spinnaker_mf_spikes = convert_spikes(MF_spikes)

print("Plotting spiking raster plot for MF")
f, axes = plt.subplots(1, 1,
                       figsize=(6, 6), sharex=True, dpi=400)

# spike raster
_times = spinnaker_mf_spikes[:, 1]
_ids = spinnaker_mf_spikes[:, 0]
axes.scatter(_times, _ids, s=.5, rasterized=True)
axes.set_title(use_display_name("mf"))
plt.xlabel("Time (ms)")
f.tight_layout()
save_figure(plt, os.path.join(fig_folder, "mf_raster"),
            extensions=['.png', ])
plt.close(f)

print("Done")
