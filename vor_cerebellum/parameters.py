"""
Parameters used in the cerebellum experiments
"""

L_RATE = 2
H_RATE = 20

# Units used in PyNN simulations: http://neuralensemble.org/docs/PyNN/units.html

# Ring buffer left shifts (RB LS)
rbls = {
    'mossy_fibers': None,
    'granule': [6, 6],
    'golgi': [6, 6],
    'purkinje': [6, 6],
    'vn': [6, 6],
    'climbing_fibers': None,
}

# Neuron parameters
neuron_params = {
    "v_thresh": -50,
    "v_reset": -70,
    "v_rest": -65,
    "i_offset": 0  # DC input
}

# PF-PC learning parameters
pfpc_min_weight = 0
pfpc_max_weight = 0.01
pfpc_initial_weight = 0.01
pfpc_ltp_constant = 0.01
pfpc_t_peak = 100  # ms
pfpc_plasticity_delay = 4  # ms

# MF-VN learning parameters
mfvn_min_weight = 0
mfvn_max_weight = 0.005
mfvn_initial_weight = 0.001
mfvn_ltp_constant = 0.001
mfvn_beta = 11
mfvn_sigma = 201
mfvn_plasticity_delay = 4  # ms


# Neurons
# Network parameters
num_MF_neurons = 100
num_GC_neurons = 2000
num_GOC_neurons = 100
num_PC_neurons = 200
num_VN_neurons = 200
num_CF_neurons = 200
all_neurons = {
    "mossy_fibers": num_MF_neurons,
    "granule": num_GC_neurons,
    "golgi": num_GOC_neurons,
    "purkinje": num_PC_neurons,
    "vn": num_VN_neurons,
    "climbing_fibers": num_CF_neurons
}

# Static weights and dela
CONNECTIVITY_MAP = {
    'mf_grc': {
        'pre': 'mossy_fibers',
        'post': 'granule',
        'weight': 0.5,  # uS
        'delay': [1.0, 10.0],  # ms
    },
    'mf_goc': {
        'pre': 'mossy_fibers',
        'post': 'golgi',
        'weight': 0.1,  # uS
        'delay': [1.0, 10.0],  # ms
    },
    'mf_vn': {
        'pre': 'mossy_fibers',
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
        'weight': 0.002,  # uS
        'delay': [1.0, 10.0],  # ms
    },

    'pc_vn': {
        'pre': 'purkinje',
        'post': 'vn',
        'weight': 0.01,  # uS
        'delay': [1.0, 10.0],  # ms
    },

    'cf_pc': {
        'pre': 'climbing_fibres',
        'post': 'purkinje',
        'weight': 0.0,  # uS
        'delay': 4,  # ms
    },
}
