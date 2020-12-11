from brian2.units import nS, uS

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
pfpc_max_weight = 0.1
pfpc_initial_weight = 0.01
pfpc_ltp_constant = 0.01
pfpc_t_peak = 100  # ms
pfpc_plasticity_delay = 4  # ms

# MF-VN learning parameters
mfvn_min_weight = 0
mfvn_max_weight = 0.1
mfvn_initial_weight = 0.005
mfvn_ltp_constant = 0.01
mfvn_beta = 11
mfvn_sigma = 201
mfvn_plasticity_delay = 4  # ms


# Static weights
conn_params = {
    # TODO
}
