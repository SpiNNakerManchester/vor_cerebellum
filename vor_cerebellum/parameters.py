from brian2.units import nS, uS


# PF-PC learning parameters
pfpc_min_weight = 0
pfpc_max_weight = 10 * nS / uS
pfpc_ltp_constant = 1 * nS / uS
pfpc_t_peak = 100  # ms
pfpc_initial_weight = 4 * nS / uS
