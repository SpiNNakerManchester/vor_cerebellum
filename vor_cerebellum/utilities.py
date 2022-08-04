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
Utilities mostly taken from https://github.com/spinnakermanchester/spinncer
"""

import numpy as np
import pylab as plt
import matplotlib as mlib
import copy
import os
import string
import traceback
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable  # ImageGrid

from elephant.spike_train_generation import homogeneous_poisson_process
import quantities as pq

mlib.use('Agg')
# ensure we use viridis as the default cmap
plt.viridis()

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})
viridis_cmap = mlib.cm.get_cmap('viridis')

ICUB_VOR_VENV_POP_SIZE = 2
POS_TO_VEL = 2 * np.pi * 0.001

fig_folder = "figures/"
# Check if the folders exist
if not os.path.exists(fig_folder):
    os.mkdir(fig_folder)

result_dir = "results/"
# Check if the results folder exist
if not os.path.exists(result_dir):
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
    'stim_radius': "Stimulation radius ($mu m$)",
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
    'vn': 'VN',
    'grc': 'GrC',
    'goc': 'GoC',
    'cf': 'CF',
    'mf': 'MF',
    'pc': 'PC',
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
def get_error(icub_vor_env_pop):
    b_vertex = icub_vor_env_pop._vertex
    error = b_vertex.get_data('error')
    return error.tolist()


def get_l_count(icub_vor_env_pop):
    b_vertex = icub_vor_env_pop._vertex
    left_count = b_vertex.get_data('l_count')
    return left_count.tolist()


def get_r_count(icub_vor_env_pop):
    b_vertex = icub_vor_env_pop._vertex
    right_count = b_vertex.get_data('r_count')
    return right_count.tolist()


def get_eye_pos(icub_vor_env_pop):
    b_vertex = icub_vor_env_pop._vertex
    eye_positions = b_vertex.get_data('eye_pos')
    return eye_positions.tolist()


def get_eye_vel(icub_vor_env_pop):
    b_vertex = icub_vor_env_pop._vertex
    eye_velocities = b_vertex.get_data('eye_vel')
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

    vel = np.diff(temp_pos) / (POS_TO_VEL * slowdown)

    vel = np.repeat(vel[::int(slowdown)], slowdown)

    # if slowdown == 1:
    #     assert np.all(np.isclose(vel, -np.cos(i * 2 * np.pi)), 0.001)

    return pos, vel


def retrieve_and_package_results(icub_vor_env_pop):
    # Get the data from the ICubVorEnv pop
    errors = np.asarray(get_error(icub_vor_env_pop=icub_vor_env_pop)).ravel()
    l_counts = get_l_count(icub_vor_env_pop=icub_vor_env_pop)
    r_counts = get_r_count(icub_vor_env_pop=icub_vor_env_pop)
    rec_eye_pos = np.asarray(get_eye_pos(
        icub_vor_env_pop=icub_vor_env_pop)).ravel()
    rec_eye_vel = np.asarray(get_eye_vel(
        icub_vor_env_pop=icub_vor_env_pop)).ravel()
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


def plot_results(
        results_dict, simulation_parameters, name, all_spikes, xlim=None):
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
    pc_spikes = all_spikes['purkinje']
    perfect_eye_pos = simulation_parameters['perfect_eye_pos']
    perfect_eye_vel = simulation_parameters['perfect_eye_vel']
    vn_size = simulation_parameters['vn_size']
    cf_size = simulation_parameters['cf_size']

    # plot the data from the ICubVorEnv pop
    x_plot = np.array([(n) for n in range(0, runtime, error_window_size)])
    fig = plt.figure(figsize=(15, 20), dpi=400)
    # Spike raster plot
    ax = plt.subplot(6, 1, 1)
    highlight_area(ax, runtime, vn_size // 2, vn_size)
    first_half_filter = vn_spikes[:, 0] < vn_size // 2
    second_half_filter = ~first_half_filter
    plt.scatter(
        vn_spikes[second_half_filter, 1], vn_spikes[second_half_filter, 0],
        s=1, color=viridis_cmap(.75), rasterized=True)
    plt.scatter(
        vn_spikes[first_half_filter, 1], vn_spikes[first_half_filter, 0],
        s=1, color=viridis_cmap(.25), rasterized=True)
    ax.set_ylabel("VN")
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    plt.ylim([-0.1, vn_size + 0.1])
    # L/R counts
    ax2 = plt.subplot(6, 1, 2)
    plt.plot(x_plot, l_counts, 'o', color=viridis_cmap(.25), label="l_counts",
             rasterized=True)
    plt.plot(x_plot, r_counts, 'o', color=viridis_cmap(.75), label="r_counts",
             rasterized=True)
    plt.legend(loc="best")
    ax2.set_ylabel("R/L accums.")
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    # Positions and velocities

    len_recs = len(rec_eye_pos.ravel())
    # Pos and vel
    ax2 = plt.subplot(6, 1, 3)
    plt.plot(x_plot, rec_eye_pos, label="rec. eye position")
    plt.plot(x_plot, rec_eye_vel, label="rec. eye velocity")
    plt.plot(x_plot, np.tile(
        perfect_eye_pos[::error_window_size], runtime // 1000)[:len_recs],
             label="eye position", ls=':', rasterized=True)
    plt.plot(x_plot, np.tile(
        perfect_eye_vel[::error_window_size], runtime // 1000)[:len_recs],
             label="eye velocity", ls=':', rasterized=True)
    plt.legend(loc="best")
    ax2.set_ylabel("Pos. & Vel.")
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    # Errors
    ax2 = plt.subplot(6, 1, 4)
    plt.plot(x_plot, errors, label="recorded error")
    eye_pos_diff = np.tile(
        perfect_eye_pos[::error_window_size], runtime // 1000)[
            :len_recs] - rec_eye_pos.ravel()
    eye_vel_diff = np.tile(
        perfect_eye_vel[::error_window_size], runtime // 1000)[
            :len_recs] - rec_eye_vel.ravel()
    # reconstructed_error = eye_pos_diff + eye_vel_diff

    # plt.plot(x_plot, reconstructed_error, color='k', ls=":",
    #          label="reconstructed error")
    plt.plot(x_plot, eye_pos_diff,
             label="eye position diff", rasterized=True)
    plt.plot(x_plot, eye_vel_diff,
             label="eye velocity diff", rasterized=True)
    plt.legend(loc="best")
    ax2.set_ylabel("Error")
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    # Error spikes
    ax2 = plt.subplot(6, 1, 5)
    highlight_area(ax2, runtime, cf_size // 2, cf_size)
    first_half_filter = cf_spikes[:, 0] < cf_size // 2
    second_half_filter = ~first_half_filter
    plt.scatter(
        cf_spikes[second_half_filter, 1], cf_spikes[second_half_filter, 0],
        s=1, color=viridis_cmap(.75), rasterized=True)
    plt.scatter(
        cf_spikes[first_half_filter, 1], cf_spikes[first_half_filter, 0],
        s=1, color=viridis_cmap(.25), rasterized=True)
    ax2.set_ylabel("CF")

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])

    # PC spikes
    ax2 = plt.subplot(6, 1, 6)
    highlight_area(ax2, runtime, cf_size // 2, cf_size)
    first_half_filter = pc_spikes[:, 0] < cf_size // 2
    second_half_filter = ~first_half_filter
    ax2.scatter(
        pc_spikes[second_half_filter, 1], pc_spikes[second_half_filter, 0],
        s=1, color=viridis_cmap(.75), rasterized=True)
    ax2.scatter(
        pc_spikes[first_half_filter, 1], pc_spikes[first_half_filter, 0],
        s=1, color=viridis_cmap(.25), rasterized=True)
    ax2.set_ylabel("PC")

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([0, runtime])
    plt.ylim([-0.1, cf_size + 0.1])
    plt.xlabel("Time (ms)")
    save_figure(plt, name, extensions=[".png", ".pdf", ])
    plt.close(fig)


# def color_for_index(index, size, cmap=viridis_cmap):
#     return cmap(1 / (size - index + 1))


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
    # Head position and velocity seem to be retrieved from a look-up table
    # then updated
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

    # generate gaussian distributions around the neuron tuned to a given
    # sensorial activity
    for i in range(50):
        mean = float(i) / 50.0 + 0.01
        gaussian = np.exp(
            -((head_pos - mean) * (head_pos - mean)) / (2.0 * sigma * sigma))
        MF_pos_activity[i] = min_rate + gaussian * (max_rate - min_rate)

    for i in range(50):
        mean = float(i) / 50.0 + 0.01
        gaussian = np.exp(
            -((head_vel - mean) * (head_vel - mean)) / (2.0 * sigma * sigma))
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
    # rate = []
    # for i in range (100):
    #     if(i < up_neuron_ID_threshold):
    #         rate.append(max_rate)
    #     elif(i<low_neuron_ID_threshold):
    #         aux_rate=max_rate - (max_rate-min_rate)*(
    #             (i - up_neuron_ID_threshold)/(
    #                 low_neuron_ID_threshold - up_neuron_ID_threshold))
    #         rate.append(aux_rate)
    #     else:
    #         rate.append(min_rate)
    #
    # if error_>=0.0:
    #     IO_agonist[0:100]=min_rate
    #     IO_antagonist=rate
    # else:
    #     IO_antagonist[0:100]=min_rate
    #     IO_agonist=rate
    IO_agonist[:] = high_rate
    IO_antagonist[:] = low_rate

    ea_rate = np.concatenate((IO_agonist, IO_antagonist))

    return ea_rate


def process_VN_spiketrains(VN_spikes, t_start):
    total_spikes = 0
    for spiketrain in VN_spikes.segments[0].spiketrains:
        s = spiketrain.as_array(
            )[np.where(spiketrain.as_array() >= t_start)[0]]
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
    for i, rate, start, duration in zip(
            range(n_inputs), rates, starts, durations):
        curr_spikes = []
        for r, s, d in zip(rate, start, duration):
            curr_spikes.append(homogeneous_poisson_process(
                rate=r * pq.Hz,
                t_start=s * pq.ms,
                t_stop=(s + d) * pq.ms,
                as_array=True))
        spike_times[i] = np.concatenate(curr_spikes)
    return spike_times


def floor_spike_time(
        times, dt=0.1 * pq.ms, t_start=0 * pq.ms, t_stop=1000.0 * pq.ms):
    bins = np.arange(t_start, t_stop + dt, dt)
    count, bin_edges = np.histogram(times, bins=bins)
    present_times_filter = count > 0
    selected_spike_times = (bin_edges[:-1])[present_times_filter]
    # Allow for multiple spikes in a timestep if that's how times get rounded
    rounded_spike_times = np.repeat(
        selected_spike_times, repeats=count[present_times_filter])
    # Check that there are the same number of spikes out as spikes in
    assert (len(rounded_spike_times) == len(times))
    return rounded_spike_times


def ff_1_to_1_odd_even_mapping(no_nids):
    sources = np.arange(no_nids)
    targets = np.ones(no_nids) * np.nan
    targets[:no_nids // 2] = sources[:no_nids // 2] * 2
    targets[no_nids // 2:] = ((sources[no_nids // 2:] - no_nids // 2) * 2) + 1
    return np.vstack((sources, targets[::])).T


def ff_1_to_1_odd_even_mapping_reversed(no_nids):
    sources = np.arange(no_nids)
    targets = np.ones(no_nids) * np.nan
    targets[:no_nids // 2] = ((sources[no_nids // 2:] - no_nids // 2) * 2) + 1
    targets[no_nids // 2:] = sources[:no_nids // 2] * 2
    return np.vstack((sources, targets[::])).T


def enable_recordings_for(populations, full_recordings=False):
    # Records spikes on spikes sources
    populations["mossy_fibres"].record(['spikes'])
    populations["climbing_fibres"].record(['spikes'])

    # Record things for LIF populations
    if full_recordings:
        # Record everything
        populations["granule"].record('all')
        populations["golgi"].record('all')
        populations["vn"].record('all')
        populations["purkinje"].record('all')
    else:
        # Record just the spikes (and maybe the packets per timestep too
        populations["granule"].record(['spikes', 'packets-per-timestep'])
        populations["golgi"].record(['spikes', 'packets-per-timestep'])
        populations["vn"].record(['spikes', 'packets-per-timestep'])
        populations["purkinje"].record(['spikes', 'packets-per-timestep'])


def take_connectivity_snapshot(all_projections, final_connectivity):
    # Retrieve final network connectivity
    try:
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
            final_connectivity[label].append(conn)
    except Exception:
        # This simulator might not support the way this is done
        final_connectivity = []
        traceback.print_exc()


def retrieve_all_spikes(all_populations):
    all_spikes = {}
    for label, pop in all_populations.items():
        if pop is not None:
            print("Retrieving recordings for ", label, "...")
            all_spikes[label] = pop.get_data(['spikes'])
    return all_spikes


def retrieve_all_other_recordings(all_populations, full_recordings):
    neo_all_recordings = {}
    other_recordings = {}
    for label, pop in all_populations.items():
        if label in ["mossy_fibres", "climbing_fibres"]:
            continue
        print("Retrieving recordings for ", label, "...")
        other_recordings[label] = {}

        if full_recordings:
            other_recordings[label]['gsyn_inh'] = np.array(
                pop.get_data(['gsyn_inh']).filter(name='gsyn_inh'))[0].T
            other_recordings[label]['gsyn_exc'] = np.array(
                pop.get_data(['gsyn_exc']).filter(name='gsyn_exc'))[0].T
            other_recordings[label]['v'] = np.array(
                pop.get_data(['v']).segments[0].filter(name='v'))[0].T

        try:
            other_recordings[label]['packets'] = np.array(
                pop.get_data(['packets-per-timestep']).segments[0].filter(
                    name='packets-per-timestep'))[0].T
        except Exception:
            print("Failed to retrieve packets-per-timestep")

        neo_all_recordings[label] = pop.get_data(  # 'all')
            ['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])

    return other_recordings, neo_all_recordings


def analyse_run(results_file, fig_folder, suffix):
    # Check if the folders exist
    if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    # Get npz archive
    previous_run_data = np.load(results_file + ".npz", allow_pickle=True)

    # Get required parameters out of the
    all_spikes = previous_run_data['all_spikes'].ravel()[0]
    all_neurons = previous_run_data['all_neurons'].ravel()[0]
    # TODO make sure all new runs save this info
    # all_neuron_params = previous_run_data['all_neuron_params'].ravel()[0]
    simulation_parameters = previous_run_data[
        'simulation_parameters'].ravel()[0]
    # other_recordings = previous_run_data['other_recordings'].ravel()[0]
    final_connectivity = previous_run_data['final_connectivity'].ravel()[0]
    # simtime = previous_run_data['simtime']
    # cell_params = previous_run_data['cell_params'].ravel()[0]
    # per_pop_neurons_per_core_constraint = previous_run_data[
    #     'per_pop_neurons_per_core_constraint'].ravel()[0]
    icub_snapshots = previous_run_data['icub_snapshots']
    results = icub_snapshots[-1]
    runtime = simulation_parameters['runtime']

    # Compute plot order
    plot_order = get_plot_order(all_spikes.keys())
    n_plots = float(len(plot_order))

    # Report useful parameters
    print("=" * 80)
    print("Analysis report")
    print("-" * 80)
    print("Current time",
          datetime.now().strftime("%H:%M:%S on %d.%m.%Y"))
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
        conn = np.asarray(final_connectivity[key][-1])
        if final_connectivity[key] is None or conn.size == 0:
            print("Skipping analysing connection", key)
            continue
        # conn_exists = True
        if len(conn.shape) == 1 or conn.shape[1] != 4:
            try:
                x = np.concatenate(conn)
                conn = x
            except Exception:
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
        try:
            conn = conn_dict[key][-1]
            mean = np.mean(conn[:, 3])
            print("{:27} -> {:4.2f} ms".format(
                key, mean))
        except Exception:
            print(key)
            traceback.print_exc()

    print("Plotting spiking raster plot for all populations")
    f, axes = plt.subplots(len(all_spikes.keys()), 1,
                           figsize=(14, 20), sharex=True, dpi=400)
    for index, pop in enumerate(plot_order):
        curr_ax = axes[index]
        # spike raster
        if pop == "vn":
            _times = simulation_parameters['vn_spikes'][:, 1]
            _ids = simulation_parameters['vn_spikes'][:, 0]
        else:
            _times = all_spikes[pop][:, 1]
            _ids = all_spikes[pop][:, 0]

        curr_ax.scatter(_times,
                        _ids,
                        color=viridis_cmap(index / (n_plots + 1)),
                        s=.5, rasterized=True)
        curr_ax.set_title(use_display_name(pop))
    plt.xlabel("Time (ms)")
    f.tight_layout()
    save_figure(plt, os.path.join(fig_folder, "raster_plots" + suffix),
                extensions=['.png', '.pdf'])
    plt.close(f)

    for proj, conn in final_connectivity.items():
        if proj in ['mf_vn', 'pf_pc']:
            no_cons = len(conn)
        else:
            no_cons = 1
        for i in range(no_cons):
            f = plt.figure(1, figsize=(9, 9), dpi=400)
            plt.hist(conn[i][:, 2], bins=20)
            plt.title(use_display_name(proj))
            plt.xlabel("Weight")
            plt.ylabel("Count")
            save_figure(plt, os.path.join(
                fig_folder,
                "{}_weight_histogram_snap{}".format(proj, i) + suffix),
                        extensions=['.png', ])
            plt.close(f)

    # plot the data from the ICubVorEnv pop
    plot_results(results_dict=results,
                 simulation_parameters=simulation_parameters,
                 name=os.path.join(
                     fig_folder, "cerebellum_icub_first_1k" + suffix),
                 all_spikes=all_spikes,
                 xlim=[0, 1000])

    plot_results(results_dict=results,
                 simulation_parameters=simulation_parameters,
                 name=os.path.join(
                     fig_folder, "cerebellum_icub_last_1k" + suffix),
                 all_spikes=all_spikes,
                 xlim=[runtime - 1000, runtime])

    plot_results(results_dict=results,
                 simulation_parameters=simulation_parameters,
                 all_spikes=all_spikes,
                 name=os.path.join(
                     fig_folder, "cerebellum_icub_full" + suffix))

    # unpacking results
    errors = icub_snapshots[i]['errors']
    # l_counts = np.asarray(icub_snapshots[i]['l_counts']).ravel()
    # r_counts = np.asarray(icub_snapshots[i]['r_counts']).ravel()
    # rec_eye_pos = icub_snapshots[i]['rec_eye_pos']
    # rec_eye_vel = icub_snapshots[i]['rec_eye_vel']

    # unpacking simulation params
    runtime = simulation_parameters['runtime']
    error_window_size = simulation_parameters['error_window_size']
    # vn_spikes = simulation_parameters['vn_spikes']
    # cf_spikes = simulation_parameters['cf_spikes']
    perfect_eye_pos = simulation_parameters['perfect_eye_pos']
    # perfect_eye_vel = simulation_parameters['perfect_eye_vel']
    # vn_size = simulation_parameters['vn_size']
    # cf_size = simulation_parameters['cf_size']

    # Error plots
    # Evolution of error
    # print("Plotting boxplot for {} for all population".format(variable_name))
    pattern_period = perfect_eye_pos.shape[0]  # in ms

    n_plots = runtime / (pattern_period)
    error_windows_per_pattern = int(pattern_period / error_window_size)
    reshaped_error = errors.reshape(errors.size // error_windows_per_pattern,
                                    error_windows_per_pattern)
    # maes = np.mean(np.abs(reshaped_error), axis=1)

    bp_width = 0.7
    f = plt.figure(figsize=(12, 8), dpi=600)
    for index in np.arange(reshaped_error.shape[0]):
        curr_data = reshaped_error[index]
        plt.boxplot(curr_data, notch=True, positions=[index + 1],
                    medianprops=dict(
                        color=color_for_index(index, n_plots),
                        linewidth=1.5),
                    widths=bp_width
                    )
    plt.ylabel("Error")
    plt.xlim([0, n_plots + 1])
    plt.grid(True, which="major", axis="y")
    plt.xlabel('Pattern #')
    xtick_display_names = [
        str(int(x)) for x in np.arange(reshaped_error.shape[0]) + 1]
    _, _labels = plt.xticks(np.arange(n_plots) + 1, xtick_display_names)

    f.tight_layout()
    save_figure(plt, os.path.join(fig_folder,
                                  "bp_errors{}".format(suffix)),
                extensions=['.png', '.pdf'])
    plt.close(f)

    # Plot at 3 times during the simulation
    f, axes = plt.subplots(
        1, 3, figsize=(14, 10), sharey='row', sharex='row', dpi=400)
    periods = [0, reshaped_error.shape[0] // 2, reshaped_error.shape[0] - 1]

    for index, curr_ax in enumerate(axes):
        curr_errors = reshaped_error[periods[index]]
        curr_ax.hist(curr_errors,
                     color=viridis_cmap(index / (3 + 1)),
                     bins=21, rasterized=True, orientation='horizontal')
        if index == 0:
            curr_ax.set_ylabel("Error")
        if index == 1:
            curr_ax.set_title("Error evolution")
        curr_ax.set_xlabel("Count")

    plt.tight_layout()
    save_figure(
        plt,
        os.path.join(fig_folder, "error_evolution{}".format(suffix)),
        extensions=[".png", ".pdf"])
    plt.close(f)

    f = plt.figure(figsize=(10, 7), dpi=400)
    plt.plot(np.arange(
        reshaped_error.shape[0]) + 1, np.std(reshaped_error, axis=1))
    plt.ylabel("Error std.")
    plt.xlim([0, n_plots + 1])
    plt.grid(True, which="major", axis="y")
    plt.xlabel('Pattern #')
    xtick_display_names = [
        str(int(x)) for x in np.arange(reshaped_error.shape[0]) + 1]
    _, _labels = plt.xticks(np.arange(n_plots) + 1, xtick_display_names)

    f.tight_layout()
    save_figure(plt, os.path.join(fig_folder,
                                  "error_std{}".format(suffix)),
                extensions=['.png', '.pdf'])
    plt.close(f)

    f = plt.figure(figsize=(10, 7), dpi=400)
    plt.plot(np.arange(
        reshaped_error.shape[0]) + 1, np.mean(np.abs(reshaped_error), axis=1))
    plt.ylabel("Abs. error")
    plt.xlim([0, n_plots + 1])
    plt.grid(True, which="major", axis="y")
    plt.xlabel('Pattern #')
    xtick_display_names = [
        str(int(x)) for x in np.arange(reshaped_error.shape[0]) + 1]
    _, _labels = plt.xticks(np.arange(n_plots) + 1, xtick_display_names)

    f.tight_layout()
    save_figure(plt, os.path.join(fig_folder,
                                  "error_mae{}".format(suffix)),
                extensions=['.png', '.pdf'])
    plt.close(f)

    # Looking at weights
    write_header("Looking at weights to see if they are being optimised well")
    for proj, pre_pop, post_pop in zip(['pf_pc', 'mf_vn'],
                                       ['granule', 'mossy_fibres'],
                                       ['purkinje', 'vn']):
        write_value("proj", proj)
        write_value("pre_pop", pre_pop)
        write_value("post_pop", post_pop)
        ff_conn = final_connectivity[proj][-1]
        conn_matrix = np.ones(
            (all_neurons[pre_pop], all_neurons[post_pop])) * np.nan
        delay_matrix = np.ones(conn_matrix.shape) * np.nan
        for s, t, w, d in ff_conn:
            s = int(s)
            t = int(t)
            conn_matrix[s, t] = w
            delay_matrix[s, t] = d

        # Weight matrix
        f = plt.figure(1, figsize=(12, 9), dpi=500)
        im = plt.imshow(conn_matrix,
                        interpolation='none',
                        extent=[0, all_neurons[post_pop],
                                0, all_neurons[pre_pop]],
                        origin='lower')
        ax = plt.gca()
        ax.set_aspect('auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Syn. weight (uS)")

        ax.set_xlabel("Target Neuron ID")
        ax.set_ylabel("Source Neuron ID")
        ax.set_title(use_display_name(proj))
        save_figure(plt, os.path.join(
            fig_folder, "{}_weight_matrix{}".format(proj, suffix)),
                    extensions=['.png', '.pdf'])
        plt.close(f)

        # Delay matrix
        # f = plt.figure(1, figsize=(12, 9), dpi=500)
        # im = plt.imshow(delay_matrix * 1000,
        #                 interpolation='none',
        #                 extent=[0, all_neurons[pre_pop],
        #                         0, all_neurons[post_pop]],
        #                 origin='lower')
        # ax = plt.gca()
        # ax.set_aspect('auto')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", "5%", pad="3%")
        # cbar = plt.colorbar(im, cax=cax)
        # cbar.set_label("Delay (ms)")
        #
        # ax.set_xlabel("Target Neuron ID")
        # ax.set_ylabel("Source Neuron ID")
        # plt.title(use_display_name(proj))
        # save_figure(plt, os.path.join(
        #     fig_folder, "{}_delay_matrix{}".format(proj, suffix)),
        #             extensions=['.png', '.pdf'])
        # plt.close(f)

        # Weight histograms for each side
        pop_split = conn_matrix.shape[1] // 2
        weights_per_mc = []
        for pop_i in range(2):
            weights_per_mc.append(conn_matrix[
                :, pop_i * pop_split:(pop_i + 1) * pop_split].ravel())

            print("Mean weight for this side:", np.mean(weights_per_mc[-1]))

        # For all to all connectivity this should hold
        if np.all(np.isfinite(conn_matrix)):
            assert weights_per_mc[0].size == weights_per_mc[1].size

        f = plt.figure(1, figsize=(9, 9), dpi=400)
        plt.hist(weights_per_mc, bins=20)
        plt.title(use_display_name(proj))
        plt.xlabel("Weight (uS)")
        plt.ylabel("Count")
        save_figure(plt, os.path.join(
            fig_folder, "LR_{}_weight_histogram_snap{}".format(proj, suffix)),
                    extensions=['.png', ])
        plt.close(f)
        write_sep()


def build_network_for_training():
    pass


def build_network_for_testing():
    pass
