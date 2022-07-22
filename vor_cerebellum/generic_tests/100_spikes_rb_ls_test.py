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
This script tests whether spike counting additional provenance is correct
"""
try:
    # this might be deprecated soon
    import pyNN.spiNNaker as sim
except ImportError:
    import pyNN.spynnaker as sim
sim.setup(timestep=1, min_delay=1, max_delay=1)

# Compute 100 spike
n_neurons = 100
spike_times = [[314] for _ in range(n_neurons)]
# Compute 1:1 connectivity
conn = []
for i in range(n_neurons):
    conn.append((i, i))

# input pop
inp_pop = sim.Population(
    n_neurons,
    cellclass=sim.SpikeSourceArray,
    cellparams={'spike_times': spike_times},
    label="100 spike pop")

# lif pop
lif_pop = sim.Population(
    100,
    cellclass=sim.IF_curr_exp, additional_parameters={"rb_left_shifts": [1, 9]},
    label="LIF pop 100 spikes")

# connection
sim.Projection(inp_pop, lif_pop, sim.FromListConnector(conn),
               synapse_type=sim.StaticSynapse(weight=1.0, delay=1.0))

sim.run(1000)
sim.end()
