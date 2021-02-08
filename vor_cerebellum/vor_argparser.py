import argparse

DEFAULT_FIGURE_DIR = 'figures/'
DEFAULT_RESULT_DIR = 'results/'
DEFAULT_SIMTIME = 100000  # ms
DEFAULT_SINGLE_SIMTIME = 10000  # ms
DEFAULT_TIMESTEP = 1.0  # ms
DEFAULT_ERROR_WINDOW_SIZE = 10  # ms
DEFAULT_GAIN = 20
DEFAULT_MOVEMENT_SLOWDOWN_FACTOR = 1

# rates used by cf
DEFAULT_BACKGROUND_RATE = 2.  # Hz
DEFAULT_BURST_RATE = 20.  # Hz

# Simulator
DEFAULT_SIMULATOR = "spinnaker"

# Scale weights from the default
DEFAULT_WEIGHT_SCALING = 1.

DEFAULT_TIMESCALE = None

DEFAULT_NEST_GRID_MODE = 'off_grid'

parser = argparse.ArgumentParser(
    description='Run a cerebellar simulation written in PyNN '
                'using sPyNNaker on SpiNNaker or using NEST on (H)PC.',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--simtime', type=int,
                    help="total simulation time (in ms) "
                         "-- default {} ms".format(DEFAULT_SINGLE_SIMTIME),
                    default=DEFAULT_SINGLE_SIMTIME)

parser.add_argument('--gain', type=float,
                    help="amplification to contribution of a single spike to eye movement "
                         "-- default {} ms".format(DEFAULT_GAIN),
                    default=DEFAULT_GAIN)

parser.add_argument('--slowdown_factor', type=int,
                    help="slowdown factor for head movement "
                         "-- default {} ms".format(DEFAULT_MOVEMENT_SLOWDOWN_FACTOR),
                    default=DEFAULT_MOVEMENT_SLOWDOWN_FACTOR)

parser.add_argument('--single_simtime', type=int,
                    help="single simulation time (in ms) "
                         "-- default {} ms".format(DEFAULT_SINGLE_SIMTIME),
                    default=DEFAULT_SINGLE_SIMTIME)

parser.add_argument('--error_window_size', type=float,
                    help="evidence accumulation window for spinngym env "
                         "-- default {} ms".format(DEFAULT_ERROR_WINDOW_SIZE),
                    default=DEFAULT_ERROR_WINDOW_SIZE)

parser.add_argument('--weight_scaling', type=float,
                    help="scale the weights before passing them to sPyNNaker "
                         "-- default {}".format(DEFAULT_WEIGHT_SCALING),
                    default=DEFAULT_WEIGHT_SCALING)

parser.add_argument('--simulator', type=str,
                    help="which simulator to use "
                         "-- default {} ms".format(DEFAULT_SIMULATOR),
                    default=DEFAULT_SIMULATOR)

parser.add_argument('--nest_grid', type=str,
                    help="[NEST] which solver to use -- on_grid or off_grid (precise) "
                         "-- default {} ms".format(DEFAULT_NEST_GRID_MODE),
                    default=DEFAULT_NEST_GRID_MODE)

parser.add_argument('--f_base', type=float,
                    help="background firing rate of the stimulus "
                         "-- default {}Hz".format(DEFAULT_BACKGROUND_RATE),
                    default=DEFAULT_BACKGROUND_RATE)

parser.add_argument('--f_peak', type=float,
                    help="burst firing rate of the stimulus "
                         "-- default {}Hz".format(DEFAULT_BURST_RATE),
                    default=DEFAULT_BURST_RATE)

parser.add_argument('--timestep', type=float,
                    help="simulation timestep (in ms) "
                         "-- default {}ms".format(DEFAULT_TIMESTEP),
                    default=DEFAULT_TIMESTEP)

parser.add_argument('-i', '--input', type=str,
                    help="name of the dataset storing "
                         "initial connectivity for the simulation",
                    dest='path_to_input')

parser.add_argument('-o', '--output', type=str,
                    help="name of the numpy archive (.npz) "
                         "storing simulation results",
                    dest='filename')

parser.add_argument('--figures_dir', type=str,
                    help='directory into which to save figures',
                    default=DEFAULT_FIGURE_DIR)

parser.add_argument('--result_dir', type=str,
                    help='directory into which to save simulation results',
                    default=DEFAULT_RESULT_DIR)

parser.add_argument('--timescale', type=int,
                    help='set the slowdown factor of the simulation',
                    default=DEFAULT_TIMESCALE)

parser.add_argument('--r_mem', action="store_true",
                    help='pre-multiply membrane resistance into weight and '
                         'i_offset')

parser.add_argument('--wta_decision', action="store_true",
                    help='[SpiNNGym] use WTA L/R decision')

parser.add_argument('--target_reaching', action="store_true",
                    help='set position to which eyes have to target')

parser.add_argument('--worst_case_spikes', action="store_true",
                    help='[FOR ANALYSIS] if this flag is present the expensive '
                         'process of counting number of afferent '
                         'spikes is performed.')

parser.add_argument('--full_recordings', action="store_true",
                    help='Enable the recording of v and gsyn, in addition to the default recordings:'
                         ' spikes and packets per timestep per core.')

parser.add_argument('--suffix', type=str,
                    help="extra string to identify some files"
                         "-- default {}".format(''),
                    default='')

# Learning params
parser.add_argument('--vn_cm', type=float,
                    help="CM for the VN population "
                         "-- default {}".format(None),
                    default=None)

parser.add_argument('--mfvn_scale', type=float,
                    help="level of LTD at MF-VN synapses "
                         "-- default {}".format(None),
                    default=None)

parser.add_argument('--mfvn_max_weight', type=float,
                    help="max conductance for MF-VN synapses "
                         "-- default {}".format(None),
                    default=None)

parser.add_argument('--pfpc_max_weight', type=float,
                    help="max conductance for PF-PC synapses "
                         "-- default {}".format(None),
                    default=None)

parser.add_argument('--pfpc_scale', type=float,
                    help="level of LTD at PF-PC synapses "
                         "-- default {}".format(None),
                    default=None)

parser.add_argument('--pcvn_weight', type=float,
                    help="weight of PC-VN synapses "
                         "-- default {}".format(None),
                    default=None)

parser.add_argument('--mfvn_ltp_constant', type=float,
                    help="LTP constant for MF-VN synapses "
                         "-- default {}".format(None),
                    default=None)

parser.add_argument('--pfpc_ltp_constant', type=float,
                    help="LTP constant for PF-PC synapses "
                         "-- default {}".format(None),
                    default=None)

parser.add_argument('--snapshot', type=str,
                    help="[network rebuilding] select whether to rebuild network with the _last_ "
                         "extracted weights or the weights which produced the _best_ mean average error"
                         "-- default {}".format("last"),
                    default="last")

args = parser.parse_args()
from pprint import pprint as pp

print("=" * 80)
print("Args")
pp(vars(args))
print("-" * 80)
