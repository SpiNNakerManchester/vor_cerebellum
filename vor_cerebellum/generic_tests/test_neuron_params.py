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

import sys
from vor_cerebellum.parameters import (rbls, neuron_params, CONNECTIVITY_MAP)
import numpy as np
from vor_cerebellum.utilities import create_poisson_spikes, floor_spike_time, convert_spikes
import quantities as pq
import copy
import traceback

spinnaker_sim = False
def_sim = str.lower(sys.argv[1])
if def_sim in ["spinnaker", "spynnaker"]:
    try:
        # this might be deprecated soon
        import pyNN.spiNNaker as sim
    except ImportError:
        import pyNN.spynnaker as sim
    spinnaker_sim = True
elif def_sim in ["nest"]:
    import pyNN.nest as sim
else:
    raise ValueError("Simulator " + def_sim +
                     "unrecognised!")

runtime = 1000

L_RATE = 50
H_RATE = 600

num_neurons = 20

np.random.seed(3141592653)
l_rates = create_poisson_spikes(num_neurons,
                                [[L_RATE], ] * num_neurons,
                                [[0], ] * num_neurons,
                                [[runtime], ] * num_neurons)

# Round spike times to time step boundary
for id, exc_s in enumerate(l_rates):
    rounded_spike_times = floor_spike_time(exc_s, dt=1.0 * pq.ms,
                                           t_start=0 * pq.ms, t_stop=runtime * pq.ms)
    # DEALING WITH nest.lib.hl_api_exceptions.NESTErrors.BadProperty:
    # ("BadProperty in SetStatus_id: Setting status of a
    # 'spike_generator' with GID 855: spike time cannot be
    # set to 0.", 'BadProperty',
    # <SLILiteral: SetStatus_id>, ": Setting status of a
    # 'spike_generator' with GID 855: spike time cannot be set to 0.")
    # Which means IT CAN'T BE 0.1, NOT 0
    rounded_spike_times[rounded_spike_times < 2.0] = 2.0
    l_rates[id] = rounded_spike_times

h_rates = create_poisson_spikes(num_neurons,
                                [[H_RATE], ] * num_neurons,
                                [[0], ] * num_neurons,
                                [[runtime], ] * num_neurons)

# Round spike times to time step boundary
for id, exc_s in enumerate(h_rates):
    rounded_spike_times = floor_spike_time(exc_s, dt=1.0 * pq.ms,
                                           t_start=0 * pq.ms, t_stop=runtime * pq.ms)
    # Same as before
    rounded_spike_times[rounded_spike_times < 2.0] = 2.0
    h_rates[id] = rounded_spike_times

test_rates = [l_rates, h_rates]
test_case_names = ["Low", "High"]
cases = [0, 1]

typical_pop_dict = {k: None for k in CONNECTIVITY_MAP.keys()}
cases_dict = {k: copy.deepcopy(typical_pop_dict) for k in cases}

all_populations = copy.deepcopy(cases_dict)
all_projections = copy.deepcopy(cases_dict)
all_connections = copy.deepcopy(cases_dict)
all_recordings = copy.deepcopy(cases_dict)
all_spikes = copy.deepcopy(cases_dict)
recorded_spikes = copy.deepcopy(cases_dict)
recorded_voltage = copy.deepcopy(cases_dict)
recorded_gsyn_exc = copy.deepcopy(cases_dict)
recorded_gsyn_inh = copy.deepcopy(cases_dict)
recorded_no_packets = copy.deepcopy(cases_dict)

input_spikes_by_case = {k: {} for k in cases}

sim.setup(1)
low_pop = sim.Population(num_neurons,  # number of sources
                         sim.SpikeSourceArray,  # source type
                         {'spike_times': l_rates
                          },  # source spike times
                         label="low_pop"
                         )
low_pop.record(['spikes'])

high_pop = sim.Population(num_neurons,  # number of sources
                          sim.SpikeSourceArray,  # source type
                          {'spike_times': h_rates
                           },  # source spike times
                          label="high_pop"
                          )
high_pop.record(['spikes'])

input_pops = [low_pop, high_pop]
input_pop_names = ["low", "high"]

# Create all the pops to test all of the weights in the network
for case, test_name, rates_for_test, ip in zip(cases, test_case_names, test_rates, input_pops):
    print("Current case:", test_name)
    for conn_name, params in CONNECTIVITY_MAP.items():
        print("\tCreating pop", conn_name)
        post_pop = params['post']
        weight = params['weight']
        # for case_id, (ip, ipn) in enumerate(zip(input_pops, input_pop_names)):
        if spinnaker_sim:
            x = sim.Population(1, sim.IF_cond_exp(**neuron_params), label=conn_name + "_" + test_name,
                               additional_parameters={"rb_left_shifts": rbls[post_pop]})
            x.record("all")
        else:
            x = sim.Population(1, sim.IF_cond_exp(**neuron_params), label=conn_name + "_" + test_name)
            x.record(["spikes", "v", "gsyn_exc", "gsyn_inh"])
        all_populations[case][conn_name] = x
        all_projections[case][conn_name] = \
            sim.Projection(
                ip, x,
                sim.AllToAllConnector(),
                sim.StaticSynapse(weight=np.abs(weight), delay=1.0),
                label="CONN_" + conn_name + "_" + test_name,
                receptor_type='excitatory' if weight > 0 else 'inhibitory')
        input_spikes_by_case[case][conn_name] = rates_for_test

current_error = None

# Run the simulation
try:
    sim.run(runtime)  # ms
except Exception as e:
    print("An exception occurred during execution!")
    traceback.print_exc()
    current_error = e

low_rec_spikes = low_pop.get_data('spikes')
high_rec_spikes = low_pop.get_data('spikes')

print("Retrieve recordings")
# Retrieve recordings
for case, pops_for_case in all_populations.items():
    for pop_label, pop_o in pops_for_case.items():
        print("Retrieving spikes for ", pop_label, "...")
        recorded_spikes[case][pop_label] = convert_spikes(pop_o.get_data(['spikes']))

        print("Retrieving v for ", pop_label, "...")
        recorded_voltage[case][pop_label] = np.array(
            pop_o.get_data(['v']).segments[0].filter(name='v'))[0].T

        print("Retrieving gsyn exc for ", pop_label, "...")
        recorded_gsyn_exc[case][pop_label] = np.array(
            pop_o.get_data(['gsyn_exc']).segments[0].filter(name='gsyn_exc'))[0].T

        print("Retrieving gsyn inh for ", pop_label, "...")
        recorded_gsyn_inh[case][pop_label] = np.array(
            pop_o.get_data(['gsyn_inh']).segments[0].filter(name='gsyn_inh'))[0].T

        # if spinnaker_sim return packets
        if spinnaker_sim:
            recorded_no_packets[case][pop_label] = np.array(
                pop_o.get_data(['packets-per-timestep']).segments[0].filter(name='packets-per-timestep'))[0].T

for case_label, proj_dict in all_projections.items():
    for proj_label, proj_obj in proj_dict.items():
        if proj_obj is None:
            print("Projection", proj_label, "is not implemented!")
            continue
        print("Retrieving connectivity for projection ", proj_label, "...")
        try:
            conn = np.array(proj_obj.get(('weight', 'delay'),
                                  format="list")._get_data_items())
        except Exception as e:
            print("Careful! Something happened when retrieving the "
                  "connectivity:", e, "\nRetrying...")
            conn = np.array(proj_obj.get(('weight', 'delay'), format="list"))

        conn = np.array(conn.tolist())
        all_connections[case_label][proj_label] = conn

sim.end()

np.savez_compressed(
    "results/{}_testing_neurons".format(def_sim),
    all_connections=all_connections,
    # Recordings
    # Spikes
    spikes=recorded_spikes,

    # Others
    v=recorded_voltage,
    gsyn_exc=recorded_gsyn_exc,
    gsyn_inh=recorded_gsyn_inh,
    no_packets=recorded_no_packets,

    low_rec_spikes=low_rec_spikes,
    high_rec_spikes=high_rec_spikes,

    input_spikes=input_spikes_by_case,

    # Run info
    simtime=runtime,
    cases=cases,
    case_names=test_case_names
)
