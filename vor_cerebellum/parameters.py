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
pfpc_max_weight = 10 * nS / uS
pfpc_ltp_constant = 1 * nS / uS
pfpc_t_peak = 100  # ms
pfpc_initial_weight = 4 * nS / uS
