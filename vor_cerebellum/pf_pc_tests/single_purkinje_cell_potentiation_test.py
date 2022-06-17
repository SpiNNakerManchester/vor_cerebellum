"""
LTP for PF-PC cells is done via a constant amount for each pre-synaptic spike. This scripts
tests this behaviour on SpiNNaker.
"""
import pyNN.spiNNaker as sim
from pyNN.utility.plotting import Figure, Panel
from vor_cerebellum.utilities import *
from vor_cerebellum.parameters import (pfpc_min_weight, pfpc_max_weight,
                                       pfpc_ltp_constant, pfpc_t_peak,
                                       pfpc_plasticity_delay,
                                       rbls, neuron_params)

initial_weight = 0.0

sim.setup(1)  # simulation timestep (ms)

purkinje_cell = sim.Population(1,  # number of neurons
                               sim.extra_models.IFCondExpCerebellum(**neuron_params),  # Neuron model
                               label="Purkinje Cell",
                               additional_parameters={"rb_left_shifts": rbls['purkinje']},
                               )

# Spike source to send spike via synapse
spike_times = [101, 201, 301, 401, 501, 601, 701, 801, 901]

granular_cell = sim.Population(1,  # number of sources
                               sim.SpikeSourceArray,  # source type
                               {'spike_times': spike_times},  # source spike times
                               label="src1"  # identifier
                               )

# Create projection from GC to PC
pfpc_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependencePFPC(t_peak=pfpc_t_peak),
    weight_dependence=sim.extra_models.WeightDependencePFPC(w_min=pfpc_min_weight,
                                                            w_max=pfpc_max_weight,
                                                            pot_alpha=pfpc_ltp_constant),
    weight=initial_weight, delay=pfpc_plasticity_delay)

synapse_pfpc = sim.Projection(
    granular_cell, purkinje_cell, sim.OneToOneConnector(),
    synapse_type=pfpc_plas, receptor_type="excitatory")

granular_cell.record('spikes')
purkinje_cell.record("all")

pf_weights = []
no_runs = len(spike_times)
run_length = 100
runtime = run_length * no_runs
for i in range(no_runs):
    sim.run(run_length)
    pf_weights.append(synapse_pfpc.get('weight', 'list', with_address=False))

granluar_cell_spikes = granular_cell.get_data('spikes')
purkinje_data = purkinje_cell.get_data()

# Release and clean the machine
sim.end()

pf_weights = np.asarray(pf_weights).ravel()
write_value("pf-PC weight", pf_weights)
write_value("pf-PC weight 1D diff", np.diff(pf_weights))
write_value("pf-PC LTP constant", pfpc_ltp_constant)

# Plot
fig = plt.figure(figsize=(4, 4), dpi=400)
plt.step(np.arange(no_runs) * run_length, pf_weights)
plt.xlabel("Time (ms)")
plt.ylabel("PF-PC weights (uS)")
plt.tight_layout()
plt.savefig("figures/pc_constant_ltp.png")
plt.close(fig)

F = Figure(
    # plot data for postsynaptic neuron
    Panel(granluar_cell_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, runtime)),
    Panel(purkinje_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[purkinje_cell.label], yticks=True, xlim=(0, runtime)),
    Panel(purkinje_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[purkinje_cell.label], yticks=True, xlim=(0, runtime)),
    Panel(purkinje_data.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, runtime)),
)
plt.savefig("figures/pc_potentiation_collection.png", dpi=400)

thresh = 1e-05
assert np.all(np.isclose(
    pf_weights, np.arange(no_runs) * pfpc_ltp_constant, rtol=0.0, atol=thresh)
    ), "PF-PC weights are not within {} of the correct value".format(thresh)

print("Done")
