"""
This script tests  the operation of the windowing procedure for
PF + CF on the PF-PC weights, i.e. the weights from GrC to PC.

Check provenance to verify the correct number of spikes being counted.
"""
import spynnaker8 as p
from vor_cerebellum.utilities import *
import matplotlib.pyplot as plt
import numpy as np
from vor_cerebellum.parameters import (pfpc_min_weight, pfpc_max_weight,
                                       pfpc_ltp_constant, pfpc_t_peak,
                                       rbls,
                                       pfpc_initial_weight,
                                       pfpc_ltd_constant,
                                       pc_neuron_params, scaling_factor
                                       )

runtime = 300
n_timesteps = 200

all_projections = []
all_populations = []
all_sources = []
final_connectivity = []
final_input_spikes = []
final_recordings = []

p.setup(1)  # simulation timestep (ms)
cf_spike_times = [n_timesteps]
climbing_fibre = p.Population(1,  # number of sources
                              p.SpikeSourceArray,  # source type
                              {'spike_times': cf_spike_times},  # source spike times
                              label="CF"  # identifier
                              )

final_grc_spike_time = n_timesteps + 50

grc_spike_times = []

for curr_timestep_diff in range(n_timesteps):
    purkinje_cell = p.Population(1,  # number of neurons
                                 p.extra_models.IFCondExpCerebellum(**pc_neuron_params),  # Neuron model
                                 label="PC" + str(curr_timestep_diff),
                                 additional_parameters={"rb_left_shifts": rbls['purkinje']}
                                 )
    purkinje_cell.initialize(v=pc_neuron_params['v_rest'])

    # Spike source to send spike via synapse
    pf_spike_times = [curr_timestep_diff, final_grc_spike_time]
    grc_spike_times.append(pf_spike_times)

    granular_cell = p.Population(1,  # number of sources
                                 p.SpikeSourceArray,  # source type
                                 {'spike_times': pf_spike_times},  # source spike times
                                 label="GrC" + str(curr_timestep_diff)  # identifier
                                 )
    all_sources.append(granular_cell)

    # Create projection from GC to PC
    pfpc_plas = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependencePFPC(t_peak=pfpc_t_peak,
                                                              alpha=pfpc_ltd_constant),
        weight_dependence=p.extra_models.WeightDependencePFPC(w_min=pfpc_min_weight,
                                                              w_max=pfpc_max_weight,
                                                              pot_alpha=pfpc_ltp_constant),
        weight=pfpc_initial_weight, delay=1)

    synapse_pfpc = p.Projection(
        granular_cell, purkinje_cell, p.AllToAllConnector(),
        synapse_type=pfpc_plas, receptor_type="excitatory")
    all_projections.append(synapse_pfpc)

    # Create projection from CF to PC
    synapse = p.Projection(
        climbing_fibre, purkinje_cell, p.OneToOneConnector(),
        p.StaticSynapse(weight=0.0, delay=1), receptor_type="excitatory")

    granular_cell.record("spikes")
    purkinje_cell.record("all")
    all_populations.append(purkinje_cell)

climbing_fibre.record('spikes')
p.run(runtime)

climbing_fibre_spikes = climbing_fibre.get_data('spikes')

for i in range(n_timesteps):
    final_recordings.append(all_populations[i].get_data())
    final_input_spikes.append(all_sources[i].spinnaker_get_data("spikes"))
    final_connectivity.append(all_projections[i].get('weight', 'list', with_address=False))

cf_synapse_weight = synapse.get('weight', 'list', with_address=False)

connection_strength = np.asarray(final_connectivity).ravel() / scaling_factor
grc_spikes = np.asarray(final_input_spikes)

p.end()

grc_spike_times = np.asarray(grc_spike_times)

assert np.all(grc_spike_times[:, 1] == final_grc_spike_time)
assert grc_spikes.shape[1] == 2
assert np.all(grc_spikes[:, :, 1][:, 1] == final_grc_spike_time)

write_header("pf-PC LTD Curve")
write_value("CF-PC weight", cf_synapse_weight)
write_value("Initial pf-PC weight", pfpc_initial_weight / scaling_factor)
write_value("pf-PC constant LTP", pfpc_ltp_constant / scaling_factor)
write_value("pf-PC LTD scaling constant", pfpc_ltd_constant)

voltage_matrix = np.ones((n_timesteps, runtime)) * np.nan
packet_matrix = np.ones((n_timesteps, runtime)) * np.nan
gsyn_exc_matrix = np.ones((n_timesteps, runtime)) * np.nan

for i, block in enumerate(final_recordings):
    voltage_matrix[i] = np.array(block.filter(name='v')).ravel()
    packet_matrix[i] = np.array(block.filter(name='packets-per-timestep')).ravel()
    gsyn_exc_matrix[i] = np.array(block.filter(name='gsyn_exc')).ravel()

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
                                  "pf_pc_{}".format(mat_name)),
                extensions=['.png', ])
    plt.close(f)

f = plt.figure(figsize=(12, 9), dpi=500)
plt.plot(connection_strength[::-1])
plt.xlabel("Time difference (ms)")
plt.title("PF-PC LTD curve")
plt.ylabel("Syn. Weight (uS)")
plt.savefig("figures/pf_pc_ltd_curve.png")

print("Job Complete")
