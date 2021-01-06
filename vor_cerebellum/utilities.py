"""
Utilities mostly taken from https://github.com/spinnakermanchester/spinncer
"""

import numpy as np
import pylab as plt
import matplotlib as mlib
import copy
import os
import string

from elephant.spike_train_generation import homogeneous_poisson_process
import quantities as pq

# ensure we use viridis as the default cmap
plt.viridis()

mlib.use('Agg')
# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})
viridis_cmap = mlib.cm.get_cmap('viridis')

ICUB_VOR_VENV_POP_SIZE = 2
POS_TO_VEL = 2 * np.pi * 0.001

fig_folder = "figures/"
# Check if the folders exist
if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
    os.mkdir(fig_folder)

result_dir = "results/"
# Check if the results folder exist
if not os.path.isdir(result_dir) and not os.path.exists(result_dir):
    os.mkdir(result_dir)

PREFFERED_ORDER = [
    'mossy_fibres',
    'granule',
    'golgi',
    'purkinje',
    'climbing_fibres'
    'vn'
]

COMMON_DISPLAY_NAMES = {
    'f_peak': "$f_{peak}$ (Hz)",
    'spinnaker': "SpiNNaker",
    'nest': "NEST",
    'stim_radius': "Stimulation radius ($\mu m$)",
    'glomerulus cells': "Glom",
    'glomerulus': "Glom",
    'granule cells': "GrC",
    'granule': "GrC",
    'dcn cells': "DCNC",
    'dcn': "DCNC",
    'golgi cells': "GoC",
    'golgi': "GoC",
    'purkinje cells': "PC",
    'purkinje': "PC",
    'stellate cells': "SC",
    'stellate': "SC",
    'basket cells': "BC",
    'basket': "BC",
    'max_spikes_in_a_tick': "Peak # of MC packets",
    'dumped_from_a_link': "Dropped from link",
    'send_multicast_packets': "MC packets sent",
    'max_dmas_in_a_tick': "Peak # of DMAs",
    'max_pipeline_restarts': "Peak # of pipeline restarts",
    'router_provenance': "Router",
    # Connection names
    'aa_goc': "aa-GoC",
    'aa_pc': "aa-PC",
    'bc_pc': "BC-PC",
    'gj_bc': "BC-BC",
    'gj_goc': "GoC-GoC",
    'gj_sc': "SC-SC",
    'glom_dcn': "Glom-DCNC",
    'glom_goc': "Glom-GoC",
    'glom_grc': "Glom-GrC",
    'goc_grc': "GoC-GrC",
    'pc_dcn': "PC-DCNC",
    'pf_bc': "pf-BC",
    'pf_goc': "pf-GoC",
    'pf_pc': "pf-PC",
    'pf_sc': "pf-SC",
    'sc_pc': "SC-PC",
    # iCub VOR-related
    'mossy_fibres': "MF",
    'vn': "VN",
    'climbing_fibres': "CF",
    'mf_grc': "MF-GrC",
    'pc_vn': "PC-VN",
    'mf_vn': "MF-VN",
    'cf_pc': "CF-PC",
    'mf_goc': "MF-GoC",
}


def capitalise(name):
    return string.capwords(
        " ".join(
            str.split(name, "_")
        ))


def use_display_name(name):
    name = name.lower()
    return COMMON_DISPLAY_NAMES[name] \
        if name in COMMON_DISPLAY_NAMES.keys() \
        else capitalise(name)


def get_plot_order(for_keys):
    # Compute plot order
    plot_order = []
    # only focus on keys for pops that have spikes
    key_duplicate = list(for_keys)
    key_duplicate.sort()
    for pref in PREFFERED_ORDER:
        for i, key in enumerate(key_duplicate):
            if pref in key:
                plot_order.append(key)
                key_duplicate.pop(i)

    # add remaining keys to the end of plot order
    plot_order += key_duplicate
    print("Plot order:", plot_order)
    return plot_order


def color_for_index(index, size, cmap=viridis_cmap):
    return cmap(index / (size + 1))


def convert_spiketrains(spiketrains):
    """ Converts a list of spiketrains into spynakker7 format

    :param spiketrains: List of SpikeTrains
    :rtype: nparray
    """
    if len(spiketrains) == 0:
        return np.empty(shape=(0, 2))

    neurons = np.concatenate([
        np.repeat(x.annotations['source_index'], len(x))
        for x in spiketrains])
    spikes = np.concatenate([x.magnitude for x in spiketrains])
    return np.column_stack((neurons, spikes))


def convert_spikes(neo, run=0):
    """ Extracts the spikes for run one from a Neo Object

    :param neo: neo Object including Spike Data
    :param run: Zero based index of the run to extract data for
    :type run: int
    :rtype: nparray
    """
    if len(neo.segments) <= run:
        raise ValueError(
            "Data only contains {} so unable to run {}. Note run is the "
            "zero based index.".format(len(neo.segments), run))
    return convert_spiketrains(neo.segments[run].spiketrains)


# Examples of get functions for variables
def get_error(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    error = b_vertex.get_data(
        'error', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return error.tolist()


def get_l_count(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    left_count = b_vertex.get_data(
        'l_count', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return left_count.tolist()


def get_r_count(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    right_count = b_vertex.get_data(
        'r_count', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return right_count.tolist()


def get_eye_pos(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    eye_positions = b_vertex.get_data(
        'eye_pos', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return eye_positions.tolist()


def get_eye_vel(icub_vor_env_pop, simulator):
    b_vertex = icub_vor_env_pop._vertex
    eye_velocities = b_vertex.get_data(
        'eye_vel', simulator.no_machine_time_steps, simulator.placements,
        simulator.buffer_manager, simulator.machine_time_step)
    return eye_velocities.tolist()


def generate_head_position_and_velocity(time, dt=0.001, slowdown=1):
    i = np.arange(0, time, dt)
    pos = -np.sin(i * 2 * np.pi)
    vel = -np.cos(i * 2 * np.pi)

    # repeat individual positions and velocities to slow down movement
    # DOES NOT WORK FOR VELOCITY
    if slowdown > 1:
        pos = np.repeat(pos, slowdown)
    temp_pos = np.concatenate(([pos[-1]], pos))
    vel = np.diff(temp_pos) / POS_TO_VEL

    # if slowdown == 1:
    #     assert np.all(np.isclose(vel, -np.cos(i * 2 * np.pi)), 0.001)

    return pos, vel


def retrieve_and_package_results(icub_vor_env_pop, simulator):
    # Get the data from the ICubVorEnv pop
    errors = np.asarray(get_error(icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)).ravel()
    l_counts = get_l_count(icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)
    r_counts = get_r_count(icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)
    rec_eye_pos = np.asarray(get_eye_pos(
        icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)).ravel()
    rec_eye_vel = np.asarray(get_eye_vel(
        icub_vor_env_pop=icub_vor_env_pop, simulator=simulator)).ravel()
    results = {
        'errors': errors,
        'l_counts': l_counts,
        'r_counts': r_counts,
        'rec_eye_pos': rec_eye_pos,
        'rec_eye_vel': rec_eye_vel,
    }
    return results


def highlight_area(ax, runtime, start_nid, stop_nid):
    ax.fill_between(
        [0, runtime], start_nid, stop_nid,
        color='grey', alpha=0.1,
    )


def plot_results(results_dict, simulation_parameters, name, xlim=None):
    # unpacking results
    errors = results_dict['errors']
    l_counts = results_dict['l_counts']
    r_counts = results_dict['r_counts']
    rec_eye_pos = results_dict['rec_eye_pos']
    rec_eye_vel = results_dict['rec_eye_vel']

    # unpacking simulation params
    runtime = simulation_parameters['runtime']
    error_window_size = simulation_parameters['error_window_size']
    vn_spikes = simulation_parameters['vn_spikes']
    cf_spikes = simulation_parameters['cf_spikes']
    perfect_eye_pos = simulation_parameters['perfect_eye_pos']
    perfect_eye_vel = simulation_parameters['perfect_eye_vel']
    vn_size = simulation_parameters['vn_size']
    cf_size = simulation_parameters['cf_size']

    # plot the data from the ICubVorEnv pop
    x_plot = np.array([(n) for n in range(0, runtime, error_window_size)])
    fig = plt.figure(figsize=(15, 20), dpi=400)
    # Spike raster plot
    ax = plt.subplot(5, 1, 1)
    highlight_area(ax, runtime, vn_size // 2, vn_size)
    first_half_filter = vn_spikes[:, 0] < vn_size // 2
    second_half_filter = ~first_half_filter
    plt.scatter(
        vn_spikes[second_half_filter, 1], vn_spikes[second_half_filter, 0],
        s=1, color=viridis_cmap(.75))
    plt.scatter(
        vn_spikes[first_half_filter, 1], vn_spikes[first_half_filter, 0],
        s=1, color=viridis_cmap(.25))
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    plt.ylim([-0.1, vn_size + 0.1])
    # L/R counts
    plt.subplot(5, 1, 2)
    plt.plot(x_plot, l_counts, 'o', color=viridis_cmap(.25), label="l_counts")
    plt.plot(x_plot, r_counts, 'o', color=viridis_cmap(.75), label="r_counts")
    plt.legend(loc="best")
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    # Positions and velocities

    len_recs = len(rec_eye_pos.ravel())
    plt.subplot(5, 1, 3)
    plt.plot(x_plot, rec_eye_pos, label="rec. eye position")
    plt.plot(x_plot, rec_eye_vel, label="rec. eye velocity")
    plt.plot(x_plot, np.tile(perfect_eye_pos[::error_window_size], runtime // 1000)[:len_recs],
             label="eye position", ls=':')
    plt.plot(x_plot, np.tile(perfect_eye_vel[::error_window_size], runtime // 1000)[:len_recs],
             label="eye velocity", ls=':')
    plt.legend(loc="best")
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    # Errors
    plt.subplot(5, 1, 4)
    plt.plot(x_plot, errors, label="recorded error")
    eye_pos_diff = np.tile(perfect_eye_pos[::error_window_size], runtime // 1000)[:len_recs] - rec_eye_pos.ravel()
    eye_vel_diff = np.tile(perfect_eye_vel[::error_window_size], runtime // 1000)[:len_recs] - rec_eye_vel.ravel()
    reconstructed_error = eye_pos_diff + eye_vel_diff

    plt.plot(x_plot, reconstructed_error, color='k', ls=":", label="reconstructed error")
    plt.plot(x_plot, eye_pos_diff,
             label="eye position diff")
    plt.plot(x_plot, eye_vel_diff,
             label="eye velocity diff")
    plt.legend(loc="best")
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    # Error spikes
    ax2 = plt.subplot(5, 1, 5)
    highlight_area(ax2, runtime, cf_size // 2, cf_size)
    first_half_filter = cf_spikes[:, 0] < cf_size // 2
    second_half_filter = ~first_half_filter
    plt.scatter(
        cf_spikes[second_half_filter, 1], cf_spikes[second_half_filter, 0],
        s=1, color=viridis_cmap(.75))
    plt.scatter(
        cf_spikes[first_half_filter, 1], cf_spikes[first_half_filter, 0],
        s=1, color=viridis_cmap(.25))
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    plt.ylim([-0.1, cf_size + 0.1])
    plt.xlabel("Time (ms)")
    save_figure(plt, name, extensions=[".png", ])
    plt.close(fig)


def remap_odd_even(original_spikes, size):
    remapped_spikes = copy.deepcopy(original_spikes)
    mapping = np.arange(size)
    mapping[::2] = np.arange(0, size, 2) // 2
    mapping[1::2] = size // 2 + np.arange(size - 1, 0, -2) // 2
    remapped_spikes[:, 0] = mapping[remapped_spikes[:, 0].astype(int)]
    return remapped_spikes


def remap_second_half_descending(original_spikes, size):
    remapped_spikes = copy.deepcopy(original_spikes)
    mapping = np.arange(size)
    mapping[:size // 2] = np.arange(0, size // 2, 1)
    mapping[size // 2:] = np.arange(size, size // 2, -1)
    remapped_spikes[:, 0] = mapping[remapped_spikes[:, 0].astype(int)]
    return remapped_spikes


def color_for_index(index, size, cmap=viridis_cmap):
    return cmap(index / (size + 1))


def write_sep():
    print("=" * 80)


def write_line():
    print("-" * 80)


def write_header(msg):
    write_sep()
    print(msg)
    write_line()


def write_short_msg(msg, value):
    print("{:40}:{:39}".format(msg, str(value)))


def write_value(msg, value):
    print("{:60}:{:19}".format(msg, str(value)))


def save_figure(plt, name, extensions=(".png",), **kwargs):
    for ext in extensions:
        write_short_msg("Plotting", name + ext)
        plt.savefig(name + ext, **kwargs)


def sensorial_activity(head_pos, head_vel):
    # Head position and velocity seem to be retrieve from a look-up table then updated
    # single point over time
    # head_pos = _head_pos[pt]
    # head_vel = _head_vel[pt]

    head_pos = ((head_pos + 0.8) / 1.6)
    head_vel = ((head_vel + 0.8 * 2 * np.pi) / (1.6 * 2 * np.pi))

    if head_pos > 1.0:
        head_pos = 1.0
    elif head_pos < 0.0:
        head_pos = 0.0
    if head_vel > 1.0:
        head_vel = 1.0
    elif head_vel < 0.0:
        head_vel = 0.0

    min_rate = 0.0
    max_rate = 600.0
    sigma = 0.02
    MF_pos_activity = np.zeros((50))
    MF_vel_activity = np.zeros((50))

    # generate gaussian distributions around the neuron tuned to a given sensorial activity
    for i in range(50):
        mean = float(i) / 50.0 + 0.01
        gaussian = np.exp(-((head_pos - mean) * (head_pos - mean)) / (2.0 * sigma * sigma))
        MF_pos_activity[i] = min_rate + gaussian * (max_rate - min_rate)

    for i in range(50):
        mean = float(i) / 50.0 + 0.01
        gaussian = np.exp(-((head_vel - mean) * (head_vel - mean)) / (2.0 * sigma * sigma))
        MF_vel_activity[i] = min_rate + gaussian * (max_rate - min_rate)

    sa_mean_freq = np.concatenate((MF_pos_activity, MF_vel_activity))
    out = [sa_mean_freq, head_pos, head_vel]
    return out


# Error Activity: error from eye and head encoders
def error_activity(error_, low_rate, high_rate):
    #     min_rate = 1.0
    #     max_rate = 25.0
    #
    #     low_neuron_ID_threshold = abs(error_) * 100.0
    #     up_neuron_ID_threshold = low_neuron_ID_threshold - 100.0
    IO_agonist = np.zeros((100))
    IO_antagonist = np.zeros((100))
    #
    #     rate = []
    #     for i in range (100):
    #         if(i < up_neuron_ID_threshold):
    #             rate.append(max_rate)
    #         elif(i<low_neuron_ID_threshold):
    #             aux_rate=max_rate - (max_rate-min_rate)*((i - up_neuron_ID_threshold)/(low_neuron_ID_threshold - up_neuron_ID_threshold))
    #             rate.append(aux_rate)
    #         else:
    #             rate.append(min_rate)
    #
    #     if error_>=0.0:
    #         IO_agonist[0:100]=min_rate
    #         IO_antagonist=rate
    #     else:
    #         IO_antagonist[0:100]=min_rate
    #         IO_agonist=rate
    IO_agonist[:] = high_rate
    IO_antagonist[:] = low_rate

    ea_rate = np.concatenate((IO_agonist, IO_antagonist))

    return ea_rate


def process_VN_spiketrains(VN_spikes, t_start):
    total_spikes = 0
    for spiketrain in VN_spikes.segments[0].spiketrains:
        s = spiketrain.as_array()[np.where(spiketrain.as_array() >= t_start)[0]]
        total_spikes += len(s)

    return total_spikes


def retrieve_git_commit():
    """
    See https://github.com/spinnakermanchester/spinncer
    :return:
    """
    import subprocess
    from subprocess import PIPE
    bash_command = "git rev-parse HEAD"

    try:
        # We have to use `stdout=PIPE, stderr=PIPE` instead of `text=True`
        # when using Python 3.6 and earlier. Python 3.7+ will have these QOL
        # improvements
        proc = subprocess.run(bash_command.split(),
                              stdout=PIPE, stderr=PIPE, shell=False)
        return proc.stdout
    except subprocess.CalledProcessError as e:
        print("Failed to retrieve git commit HASH-", str(e))
        return "CalledProcessError"
    except Exception as e:
        print("Failed to retrieve git commit HASH more seriously-", str(e))
        return "GeneralError"


def create_poisson_spikes(n_inputs, rates, starts, durations):
    spike_times = [[] for _ in range(n_inputs)]
    for i, rate, start, duration in zip(range(n_inputs), rates, starts, durations):
        curr_spikes = []
        for r, s, d in zip(rate, start, duration):
            curr_spikes.append(homogeneous_poisson_process(
                rate=r * pq.Hz,
                t_start=s * pq.ms,
                t_stop=(s + d) * pq.ms,
                as_array=True))
        spike_times[i] = np.concatenate(curr_spikes)
    return spike_times


def floor_spike_time(times, dt=0.1 * pq.ms, t_start=0 * pq.ms, t_stop=1000.0 * pq.ms):
    bins = np.arange(t_start, t_stop + dt, dt)
    count, bin_edges = np.histogram(times, bins=bins)
    present_times_filter = count > 0
    selected_spike_times = (bin_edges[:-1])[present_times_filter]
    # Allow for multiple spikes in a timestep if that's how spike times get rounded
    rounded_spike_times = np.repeat(selected_spike_times, repeats=count[present_times_filter])
    # Check that there are the same number of spikes out as spikes in
    assert (len(rounded_spike_times) == len(times))
    return rounded_spike_times


def ff_1_to_1_odd_even_mapping(no_nids):
    sources = np.arange(no_nids)
    targets = np.ones(no_nids) * np.nan
    targets[:no_nids // 2] = sources[:no_nids // 2] * 2
    targets[no_nids // 2:] = ((sources[no_nids // 2:] - no_nids // 2) * 2) + 1
    return np.vstack((sources, targets[::-1]))
