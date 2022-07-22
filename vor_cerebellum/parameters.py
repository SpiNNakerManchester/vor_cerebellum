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
Parameters used in the cerebellum experiments
"""

L_RATE = 2
H_RATE = 20

# Units used in PyNN simulations: http://neuralensemble.org/docs/PyNN/units.html

power = int(0)
scaling_factor = 2 ** float(power)

# Ring buffer left shifts (RB LS)
rbls = {
    'mossy_fibres': None,
    'granule': [6, 6],
    'golgi': [8, 8],
    'purkinje': [6, 6],
    'vn': [6, 6],
    'climbing_fibres': None,
}

# Neuron parameters
neuron_params = {
    "v_thresh": -50 * scaling_factor,
    "v_reset": -70 * scaling_factor,
    "v_rest": -65 * scaling_factor,
    "i_offset": 0 * scaling_factor  # DC input
}

# Neuron parameters PC
pc_neuron_params = {
    "cm": 1.0,
    "v_thresh": -50 * scaling_factor,
    "v_reset": -70 * scaling_factor,
    "v_rest": -65 * scaling_factor,
    "i_offset": 0 * scaling_factor
}

# Neuron parameters
vn_neuron_params = {
    "cm": 1.0 * scaling_factor,
    "v_thresh": -50 * scaling_factor,
    "v_reset": -70 * scaling_factor,
    "v_rest": -65 * scaling_factor,
    "i_offset": 0 * scaling_factor
}

# PF-PC learning parameters
pfpc_min_weight = 0 * scaling_factor
pfpc_max_weight = 2.9e-3 * scaling_factor  # 0.001  # setting this too high will silence VN too much.
pfpc_initial_weight = 5.0e-4 * scaling_factor  # pfpc_max_weight  # 0.0005
pfpc_ltp_constant = pfpc_max_weight / 50.  # 2**(-15 + 6) / (2**10)  # 0.00005
pfpc_ltd_constant = 0.8
pfpc_t_peak = 100  # ms
pfpc_plasticity_delay = 1.0  # ms

# MF-VN learning parameters
mfvn_min_weight = 0 * scaling_factor
mfvn_max_weight = 1.25e-3 * scaling_factor  # 0.001
mfvn_initial_weight = 1.0e-4 * scaling_factor  # 0.0005
mfvn_ltp_constant = mfvn_max_weight / 50.  # 0.0005  # 0.00005
mfvn_ltd_constant = 0.8
mfvn_beta = 11
mfvn_sigma = 201
mfvn_plasticity_delay = 1.0  # ms

# Neurons
# Network parameters
num_MF_neurons = 100
num_GC_neurons = 2000
num_GOC_neurons = 100
num_PC_neurons = 200
num_VN_neurons = 200
num_CF_neurons = 200
all_neurons = {
    "mossy_fibres": num_MF_neurons,
    "granule": num_GC_neurons,
    "golgi": num_GOC_neurons,
    "purkinje": num_PC_neurons,
    "vn": num_VN_neurons,
    "climbing_fibres": num_CF_neurons
}

# Static weights and dela
CONNECTIVITY_MAP = {
    'mf_grc': {
        'pre': 'mossy_fibres',
        'post': 'granule',
        'weight': 0.5,  # uS
        'delay': [1.0, 10.0],  # ms
    },
    'mf_goc': {
        'pre': 'mossy_fibres',
        'post': 'golgi',
        'weight': 0.1,  # uS
        'delay': [1.0, 10.0],  # ms
    },
    'mf_vn': {
        'pre': 'mossy_fibres',
        'post': 'purkinje',
        'weight': 0.0005,  # uS
        'delay': [1.0, 10.0],  # ms
    },

    'pf_pc': {
        'pre': 'granule',
        'post': 'purkinje',
        'weight': 0.005,  # uS
        'delay': [1.0, 10.0],  # ms
    },

    'goc_grc': {
        'pre': 'golgi',
        'post': 'granule',
        'weight': -0.002,  # uS
        'delay': [1.0, 10.0],  # ms
    },

    'pc_vn': {
        'pre': 'purkinje',
        'post': 'vn',
        'weight': -0.01,  # uS
        'delay': [1.0, 10.0],  # ms
    },

    'cf_pc': {
        'pre': 'climbing_fibres',
        'post': 'purkinje',
        'weight': 0.0,  # uS
        'delay': 4,  # ms
    },
}
