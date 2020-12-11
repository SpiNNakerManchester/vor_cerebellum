from __future__ import print_function
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
from vor_cerebellum.utilities import *
from vor_cerebellum.parameters import (mfvn_min_weight, mfvn_max_weight,
                                       mfvn_initial_weight,
                                       mfvn_ltp_constant,
                                       mfvn_beta, mfvn_sigma,
                                       mfvn_plasticity_delay,
                                       rbls, neuron_params)

initial_weight = 0.

p.setup(1)  # simulation timestep (ms)

vestibular_neuclei = p.Population(1,  # number of neurons
                                  p.extra_models.IFCondExpCerebellum(**neuron_params),  # Neuron model
                                  label="Vestibular Nuclei",  # identifier
                                  additional_parameters={"rb_left_shifts": rbls['vn']}
                                  )

# Spike source to send spike via synapse
spike_times = [101, 201, 301, 401, 501, 601, 701, 801, 901]

mossy_fibre_src = p.Population(1,  # number of sources
                               p.SpikeSourceArray,  # source type
                               {'spike_times': spike_times},  # source spike times
                               label="MF"  # identifier
                               )

# Create projection from MF to VN
mfvn_plas = p.STDPMechanism(
    timing_dependence=p.extra_models.TimingDependenceMFVN(beta=mfvn_beta,
                                                          sigma=mfvn_sigma),
    weight_dependence=p.extra_models.WeightDependenceMFVN(w_min=mfvn_min_weight,
                                                          w_max=mfvn_max_weight,
                                                          pot_alpha=mfvn_ltp_constant),
    weight=initial_weight, delay=mfvn_plasticity_delay)

synapse_mfvn = p.Projection(
    mossy_fibre_src, vestibular_neuclei, p.AllToAllConnector(),
    synapse_type=mfvn_plas, receptor_type="excitatory")

mossy_fibre_src.record('spikes')
vestibular_neuclei.record("all")

mf_weights = []
no_runs = len(spike_times)
run_length = 100
runtime = run_length * no_runs
for i in range(no_runs):
    p.run(run_length)
    mf_weights.append(synapse_mfvn.get('weight', 'list', with_address=False))

mossy_fibre_src_spikes = mossy_fibre_src.get_data('spikes')
vestibular_neuclei_data = vestibular_neuclei.get_data()
p.end()

mf_weights = np.asarray(mf_weights).ravel()
print(mf_weights)
print(np.diff(mf_weights))

# Plot
fig = plt.figure(figsize=(6, 6), dpi=400)
plt.step(np.arange(no_runs) * run_length, mf_weights)
plt.xlabel("Time (ms)")
plt.ylabel("MF-VN weights (uS)")
plt.tight_layout()
plt.savefig("figures/vn_constant_ltp.png")
plt.close(fig)

F = Figure(
    # plot data for postsynaptic neuron
    Panel(mossy_fibre_src_spikes.segments[0].spiketrains,
          ylabel=mossy_fibre_src.label,
          yticks=True, markersize=2, xlim=(0, runtime)),

    Panel(vestibular_neuclei_data.segments[0].filter(name='v')[0],
          ylabel="VN membrane potential (mV)",
          data_labels=[vestibular_neuclei.label], yticks=True, xlim=(0, runtime)),

    Panel(vestibular_neuclei_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="VN excitatory current (mV)",
          data_labels=[vestibular_neuclei.label], yticks=True, xlim=(0, runtime)),

    Panel(vestibular_neuclei_data.segments[0].spiketrains,
          xlabel="Time (ms)",
          yticks=True, markersize=2, xlim=(0, runtime)),
)

plt.savefig("figures/vn_single_potentiation.png", dpi=400)

thresh = 0.0001
assert np.all(np.isclose(mf_weights,
                         np.arange(no_runs) * mfvn_ltp_constant,
                         0.0001)), "MF_VN weights are not within {} of the correct value".format(thresh)

print("Done")