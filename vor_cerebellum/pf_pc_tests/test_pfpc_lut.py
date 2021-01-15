from spynnaker.pyNN.models.neuron.plasticity.stdp.common \
    import write_pfpc_lut
import math

# sin_power=20
t_peak = 100

write_pfpc_lut(spec=None,
               peak_time=t_peak,
               lut_size=256,
               shift=0,
               time_probe=t_peak,
               kernel_scaling=0.8)
