from __future__ import print_function
import spynnaker8 as sim

from pyNN.utility.plotting import Figure, Panel
from pyNN.random import RandomDistribution, NumpyRNG

# PAB imports
import traceback
import neo
# general parameters
from vor_cerebellum.parameters import (L_RATE, H_RATE, rbls, neuron_params)
# MF-VN params
from vor_cerebellum.parameters import (mfvn_min_weight, mfvn_max_weight,
                                       mfvn_initial_weight,
                                       mfvn_ltp_constant,
                                       mfvn_beta, mfvn_sigma)
# PF-PC params
from vor_cerebellum.parameters import (pfpc_min_weight, pfpc_max_weight,
                                       pfpc_initial_weight,
                                       pfpc_ltp_constant, pfpc_t_peak)
from vor_cerebellum.utilities import *

# Record SCRIPT start time (wall clock)
start_time = plt.datetime.datetime.now()

# Starting to record additional parameters


# cerebellum test bench
runtime = 1000

# Synapse parameters
gc_pc_weights = 0.005
mf_vn_weights = 0.0005
pc_vn_weights = 0.01
cf_pc_weights = 0.0
mf_gc_weights = 0.5
go_gc_weights = 0.002
mf_go_weights = 0.1

# Network parameters
num_MF_neurons = 100
num_GC_neurons = 2000
num_GOC_neurons = 100
num_PC_neurons = 200
num_VN_neurons = 200
num_CF_neurons = 200

# Random distribution for synapses delays and weights (MF and GO)
delay_distr = RandomDistribution('uniform', (1.0, 10.0), rng=NumpyRNG(seed=85524))

weight_distr_MF = RandomDistribution('uniform',
                                     (mf_gc_weights * 0.8, mf_gc_weights * 1.2),
                                     rng=NumpyRNG(seed=85524))

weight_distr_GO = RandomDistribution('uniform',
                                     (go_gc_weights * 0.8, go_gc_weights * 1.2),
                                     rng=NumpyRNG(seed=24568))

all_neurons = {
    "mossy_fibers": num_MF_neurons,
    "granule": num_GC_neurons,
    "golgi": num_GOC_neurons,
    "purkinje": num_PC_neurons,
    "vn": num_VN_neurons,
    "climbing_fibers": num_CF_neurons
}

all_populations = {

}

initial_connectivity = {

}

final_connectivity = {

}

all_projections = {

}

# Weights of pf_pc
weight_dist_pfpc = RandomDistribution('uniform',
                                      (pfpc_initial_weight * 0.8,
                                       pfpc_initial_weight * 1.2),
                                      rng=NumpyRNG(seed=24534))

global_n_neurons_per_core = 50
per_pop_neurons_per_core_constraint = {
    'mossy_fibers': global_n_neurons_per_core,
    'granule': global_n_neurons_per_core,
    'golgi': global_n_neurons_per_core,
    'purkinje': 10,
    'vn': 10,
    'climbing_fibers': global_n_neurons_per_core,
}

sim.setup(timestep=1., min_delay=1, max_delay=15)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, global_n_neurons_per_core)
sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, global_n_neurons_per_core)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp, global_n_neurons_per_core)
sim.set_number_of_neurons_per_core(sim.extra_models.IFCondExpCerebellum, global_n_neurons_per_core)

# Sensorial Activity: input activity from vestibulus (will come from the head IMU, now it is a test bench)
# We simulate the output of the head encoders with a sinusoidal function. Each "sensorial activity" value is derived from the
# head position and velocity. From that value, we generate the mean firing rate of the MF neurons (later this will be an input
# that will come from the robot, through the spinnLink)
# the neurons that are active depend on the value of the sensorial activity. For each a gaussian is created centered on a specific neuron


# Prepare variables once at beginning
MAX_AMPLITUDE = 0.8
RELATIVE_AMPLITUDE = 1.0
_head_pos = []
_head_vel = []

i = np.arange(0, 1000, 0.001)
for t in i:
    desired_speed = -np.cos(t * 2 * np.pi) * MAX_AMPLITUDE * RELATIVE_AMPLITUDE * 2.0 * np.pi
    desired_pos = -np.sin(t * 2 * np.pi) * MAX_AMPLITUDE * RELATIVE_AMPLITUDE
    _head_pos.append(desired_pos)
    _head_vel.append(desired_speed)

normalised_head_pos = (np.asarray(_head_pos) + 0.8) / 1.6
normalised_head_vel = (np.asarray(_head_vel) + 0.8 * 2 * np.pi) / (1.6 * 2 * np.pi)

###############################################################################
# ============================ Create populations ============================
##############################################################################

# Create MF population - fake input population that will be substituted by external input from robot

MF_population = sim.Population(num_MF_neurons,  # number of sources
                               sim.SpikeSourcePoisson,  # source type
                               {'rate': sensorial_activity(_head_pos[0], _head_vel[0])[0]},  # source spike times
                               label="MF",
                               additional_parameters={'seed': 24534}
                               )

all_populations["mossy_fibers"] = MF_population

# Create GOC population
GOC_population = sim.Population(num_GOC_neurons, sim.IF_cond_exp(), label='GOCLayer',
                                additional_parameters={"rb_left_shifts": rbls['golgi']}
                                )
all_populations["golgi"] = GOC_population

# create PC population
PC_population = sim.Population(num_PC_neurons,  # number of neurons
                               sim.extra_models.IFCondExpCerebellum(**neuron_params),  # Neuron model
                               label="Purkinje Cell",
                               additional_parameters={"rb_left_shifts": rbls['purkinje']}
                               )
all_populations["purkinje"] = PC_population

# create VN population
VN_population = sim.Population(num_VN_neurons,  # number of neurons
                               sim.extra_models.IFCondExpCerebellum(**neuron_params),  # Neuron model
                               label="Vestibular Nuclei",
                               additional_parameters={"rb_left_shifts": rbls['vn']}
                               )
all_populations["vn"] = VN_population

# Create GrC population
GC_population = sim.Population(num_GC_neurons, sim.IF_curr_exp(), label='GCLayer',
                               additional_parameters={"rb_left_shifts": rbls['granule']}
                               )
all_populations["granule"] = GC_population

# generate fake error (it should be calculated from sensorial activity in error activity,
# but we skip it and just generate an error from -1.5 to 1.5)
err = -0.7  # other values to test: -0.3 0 0.3 0.7

# Create CF population - fake input population that will be substituted by external input from robot
CF_population = sim.Population(num_CF_neurons,  # number of sources
                               sim.SpikeSourcePoisson,  # source type
                               {'rate': error_activity(err, L_RATE, H_RATE)},  # source spike times
                               label="CFLayer",
                               additional_parameters={'seed': 24534}
                               )
all_populations["climbing_fibers"] = CF_population

###############################################################################
# ============================ Create connections ============================
##############################################################################

# Create MF-GO connections
mf_go_connections = sim.Projection(MF_population,
                                   GOC_population,
                                   sim.OneToOneConnector(),
                                   sim.StaticSynapse(delay=delay_distr,
                                                     weight=mf_go_weights),
                                   receptor_type='excitatory')
all_projections["mf_goc"] = mf_go_connections

# Create MF-GC and GO-GC connections
float_num_MF_neurons = float(num_MF_neurons)

list_GOC_GC = []
list_MF_GC = []
list_GOC_GC_2 = []
# projections to subpopulations https://github.com/SpiNNakerManchester/sPyNNaker8/issues/168)
for i in range(num_MF_neurons):
    GC_medium_index = int(round((i / float_num_MF_neurons) * num_GC_neurons))
    GC_lower_index = GC_medium_index - 40
    GC_upper_index = GC_medium_index + 60

    if (GC_lower_index < 0):
        GC_lower_index = 0

    elif (GC_upper_index > num_GC_neurons):
        GC_upper_index = num_GC_neurons

    for j in range(GC_medium_index - GC_lower_index):
        list_GOC_GC.append(
            (i, GC_lower_index + j,
             #                  go_gc_weights, 1)
             weight_distr_GO.next(), delay_distr.next())
        )

    for j in range(GC_medium_index + 20 - GC_medium_index):
        list_MF_GC.append(
            (i, GC_medium_index + j,
             #                  mf_gc_weights, 1)
             weight_distr_MF.next(), delay_distr.next())
        )

    for j in range(GC_upper_index - GC_medium_index - 20):
        list_GOC_GC_2.append(
            (i, GC_medium_index + 20 + j,
             #                  go_gc_weights, 1)
             weight_distr_GO.next(), delay_distr.next())
        )

GO_GC_con1 = sim.Projection(GOC_population,
                            GC_population,
                            sim.FromListConnector(list_GOC_GC),
                            label='goc_grc_1',
                            receptor_type='inhibitory')  # this should be inhibitory
all_projections["goc_grc_1"] = GO_GC_con1

MF_GC_con2 = sim.Projection(MF_population,
                            GC_population,
                            sim.FromListConnector(list_MF_GC),
                            label='mf_grc',
                            receptor_type='excitatory')

all_projections["mf_grc"] = MF_GC_con2

GO_GC_con3 = sim.Projection(GOC_population,
                            GC_population,
                            sim.FromListConnector(list_GOC_GC_2),
                            label='goc_grc_2',
                            receptor_type='inhibitory')
all_projections["goc_grc_2"] = GO_GC_con3

# Create PC-VN connections
pc_vn_connections = sim.Projection(PC_population,
                                   VN_population,
                                   sim.OneToOneConnector(),
                                   label='pc_vn',
                                   # receptor_type='GABA', # Should these be inhibitory?
                                   synapse_type=sim.StaticSynapse(delay=delay_distr,
                                                                  weight=pc_vn_weights),
                                   receptor_type='inhibitory')
all_projections["pc_vn"] = pc_vn_connections

# Create MF-VN learning rule - cos

mfvn_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependenceMFVN(beta=mfvn_beta,
                                                            sigma=mfvn_sigma),
    weight_dependence=sim.extra_models.WeightDependenceMFVN(w_min=mfvn_min_weight,
                                                            w_max=mfvn_max_weight,
                                                            pot_alpha=mfvn_ltp_constant),
    weight=mfvn_initial_weight, delay=delay_distr)

# Create MF to VN connections
mf_vn_connections = sim.Projection(
    MF_population, VN_population, sim.AllToAllConnector(),  # Needs mapping as FromListConnector to make efficient
    synapse_type=mfvn_plas,
    label='mf_vn',
    receptor_type="excitatory")
all_projections["mf_vn"] = mf_vn_connections

# Create projection from PC to VN -- replaces "TEACHING SIGNAL"
pc_vn_connections = sim.Projection(
    PC_population, VN_population, sim.OneToOneConnector(),
    sim.StaticSynapse(weight=0.0, delay=1.0),
    label='pc_vn_teaching',
    receptor_type="excitatory")  # "TEACHING SIGNAL"
all_projections["pc_vn_teaching"] = pc_vn_connections

# create PF-PC learning rule - sin
pfpc_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependencePFPC(t_peak=pfpc_t_peak),
    weight_dependence=sim.extra_models.WeightDependencePFPC(w_min=pfpc_min_weight,
                                                            w_max=pfpc_max_weight,
                                                            pot_alpha=pfpc_ltp_constant),
    weight=pfpc_initial_weight, delay=delay_distr)

# Create PF-PC connections
pf_pc_connections = sim.Projection(
    GC_population, PC_population, sim.AllToAllConnector(),
    synapse_type=pfpc_plas,
    label='pf_pc',
    receptor_type="excitatory")
all_projections["pf_pc"] = pf_pc_connections

# Create IO-PC connections. This synapse with "receptor_type=COMPLEX_SPIKE" propagates the learning signals that drive the plasticity mechanisms in GC-PC synapses
cf_pc_connections = sim.Projection(CF_population,
                                   PC_population,
                                   sim.OneToOneConnector(),
                                   label='cf_pc',
                                   # receptor_type='COMPLEX_SPIKE',
                                   synapse_type=sim.StaticSynapse(delay=1.0,
                                                                  weight=cf_pc_weights),
                                   receptor_type='excitatory')
all_projections["cf_pc"] = cf_pc_connections

# ============================  Set up recordings ============================

MF_population.record(['spikes'])
CF_population.record(['spikes'])
GC_population.record('all')
# GOC_population.record(['spikes'])
GOC_population.record('all')
VN_population.record('all')  # VN_population.record(['spikes'])
# PC_population.record(['spikes'])
PC_population.record('all')

# ============================  Set up constraints ============================

for pop_name, constraint in per_pop_neurons_per_core_constraint.items():
    print("Setting NPC=", constraint, "for", pop_name)
    all_populations[pop_name].set_max_atoms_per_core(constraint)

# ============================  Run simulation ============================

samples_in_repeat = 99
sample_time = 10
repeats = 1
total_runtime = 0
VN_transfer_func = []

print("=" * 80)
print("Running simulation for", runtime, " ms split into", samples_in_repeat, "chunks.")
all_spikes_first_trial = {}
# Record simulation start time (wall clock)
sim_start_time = plt.datetime.datetime.now()
for i in range(samples_in_repeat):
    sim.run(sample_time)

    VN_spikes = VN_population.get_data('spikes')
    VN_transfer_func.append(process_VN_spiketrains(VN_spikes, total_runtime))

    total_runtime += sample_time

    print(total_runtime)

    MF_population.set(rate=sensorial_activity(_head_pos[total_runtime], _head_vel[total_runtime])[0])

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
sim_total_time = end_time - sim_start_time

total_runtime = runtime * repeats

# ============================  Retrieving data from simulation ============================

MF_spikes = MF_population.get_data('spikes')
CF_spikes = CF_population.get_data('spikes')
GC_spikes = GC_population.get_data('all')
GOC_spikes = GOC_population.get_data('spikes')
VN_spikes = VN_population.get_data('all')  # VN_population.get_data('spikes')
PC_spikes = PC_population.get_data('spikes')

mfvn_weights = mf_vn_connections.get('weight', 'list', with_address=False)
pfpc_weights = pf_pc_connections.get('weight', 'list', with_address=False)

# Retrieve recordings
all_spikes = {}
for label, pop in all_populations.items():
    if pop is not None:
        print("Retrieving recordings for ", label, "...")
        all_spikes[label] = pop.get_data(['spikes'])
other_recordings = {}
for label, pop in all_populations.items():
    if label in ["mossy_fibers", "climbing_fibers"]:
        continue
    print("Retrieving recordings for ", label, "...")
    other_recordings[label] = {}

    other_recordings[label]['current'] = np.array(
        pop.get_data(['gsyn_inh']).filter(name='gsyn_inh'))[0].T

    other_recordings[label]['gsyn'] = np.array(
        pop.get_data(['gsyn_exc']).filter(name='gsyn_exc'))[0].T

    other_recordings[label]['v'] = np.array(
        pop.get_data(['v']).segments[0].filter(name='v'))[0].T

# Retrieve final network connectivity
try:
    final_connectivity = {}
    for label, p in all_projections.items():
        if p is None:
            print("Projection", label, "is not implemented!")
            continue
        print("Retrieving connectivity for projection ", label, "...")
        try:
            conn = \
                np.array(p.get(('weight', 'delay'),
                               format="list")._get_data_items())
        except Exception as e:
            print("Careful! Something happened when retrieving the "
                  "connectivity:", e, "\nRetrying...")
            conn = \
                np.array(p.get(('weight', 'delay'), format="list"))

        conn = np.array(conn.tolist())
        final_connectivity[label] = conn
except:
    # This simulator might not support the way this is done
    final_connectivity = []
    traceback.print_exc()

sim.end()
print("job done")
# Report time taken
print("Total time elapsed -- " + str(total_time))
# ============================  Plotting some stuff ============================
# ============================  PAB ANALYSIS ============================

# Compute plot order
plot_order = get_plot_order(all_spikes.keys())
n_plots = float(len(plot_order))

# Check if using neo blocks
neo_all_spikes = {}
for pop, potential_neo_block in all_spikes.items():
    if isinstance(potential_neo_block, neo.Block):
        # make a copy of the spikes dict
        neo_all_spikes[pop] = potential_neo_block
        all_spikes[pop] = convert_spikes(potential_neo_block)

# Report useful parameters
print("=" * 80)
print("Analysis report")
print("-" * 80)
print("Current time",
      plt.datetime.datetime.now().strftime("%H:%M:%S on %d.%m.%Y"))
# Report number of neurons
print("=" * 80)
print("Number of neurons in each population")
print("-" * 80)
for pop in plot_order:
    print("\t{:20} -> {:10} neurons".format(pop, all_neurons[pop]))

# Report weights values
print("Average weight per projection")
print("-" * 80)
conn_dict = {}
for key in final_connectivity:
    # Connection holder annoyance here:
    conn = np.asarray(final_connectivity[key])
    if final_connectivity[key] is None or conn.size == 0:
        print("Skipping analysing connection", key)
        continue
    conn_exists = True
    if len(conn.shape) == 1 or conn.shape[1] != 4:
        try:
            x = np.concatenate(conn)
            conn = x
        except:
            traceback.print_exc()
        names = [('source', 'int_'),
                 ('target', 'int_'),
                 ('weight', 'float_'),
                 ('delay', 'float_')]
        useful_conn = np.zeros((conn.shape[0], 4), dtype=np.float)
        for i, (n, _) in enumerate(names):
            useful_conn[:, i] = conn[n].astype(np.float)
        final_connectivity[key] = useful_conn.astype(np.float)
        conn = useful_conn.astype(np.float)
    conn_dict[key] = conn
    mean = np.mean(conn[:, 2])
    print("{:27} -> {:4.6f} uS".format(
        key, mean))

print("Average Delay per projection")
print("-" * 80)
for key in final_connectivity:
    conn = conn_dict[key]
    mean = np.mean(conn[:, 3])
    print("{:27} -> {:4.2f} ms".format(
        key, mean))

print("Plotting spiking raster plot for all populations")
f, axes = plt.subplots(len(all_spikes.keys()), 1,
                       figsize=(14, 20), sharex=True, dpi=400)
for index, pop in enumerate(plot_order):
    curr_ax = axes[index]
    # spike raster
    _times = all_spikes[pop][:, 1]
    _ids = all_spikes[pop][:, 0]

    curr_ax.scatter(_times,
                    _ids,
                    color=viridis_cmap(index / (n_plots + 1)),
                    s=.5, rasterized=True)
    curr_ax.set_title(use_display_name(pop))
plt.xlabel("Time (ms)")
f.tight_layout()
save_figure(plt, os.path.join(fig_folder, "raster_plots"),
            extensions=['.png', '.pdf'])
plt.close(f)

for proj, conn in final_connectivity.items():
    f = plt.figure(1, figsize=(9, 9), dpi=400)
    plt.hist(conn[:, 2], bins=20)
    plt.title(use_display_name(proj))
    plt.xlabel("Weight")
    plt.ylabel("Count")
    save_figure(plt, os.path.join(fig_folder, "{}_weight_histogram".format(proj)),
                extensions=['.png', ])
    plt.close(f)

F = Figure(
    Panel(MF_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='MF_spikes'),
    Panel(GC_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='GC_spikes'),
    Panel(GOC_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='GOC_spikes'),
    Panel(PC_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='PC_spikes'),
    Panel(CF_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='CF_spikes'),
    Panel(VN_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='VN_spikes'),
    Panel(VN_spikes.segments[0].filter(name='gsyn_inh')[0],
          ylabel="Membrane potential (mV)", yticks=True, xlim=(0, total_runtime))
)
save_figure(plt, os.path.join(fig_folder, "collections"),
            extensions=['.png', ])
# plt.show(block=False)

plt.figure()
plt.plot(mfvn_weights,
         label='mf-vn weights (init: {})'.format(mfvn_initial_weight))
plt.title("mfvn_weights")
plt.legend()
save_figure(plt, os.path.join(fig_folder, "mfvn_weights"),
            extensions=['.png', ])

plt.figure()
plt.title("pfpc_weights")
plt.plot(pfpc_weights, color='orange',
         label='pf-pc weights (init: {})'.format(pfpc_initial_weight))
plt.legend()
save_figure(plt, os.path.join(fig_folder, "pfpc_weights"),
            extensions=['.png', ])

print(VN_transfer_func)
plt.figure()
plt.plot(VN_transfer_func)
plt.title("vn_transfer_function")
save_figure(plt, os.path.join(fig_folder, "VN_transfer_func"),
            extensions=['.png', ])
