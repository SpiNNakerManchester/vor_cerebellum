# Copyright (c) 2022 The University of Manchester
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

# This script assumes it is run from the directory holding all github projects in parallel
# bash SupportScripts/automatic_make.sh

do_make() {
    if [ -d "$1" ]; then
        # Control will enter here if DIRECTORY exists.
        # Run setup.bash if it exists
        if [ -f "$1/setup.bash" ]; then
            cd $1
            source setup.bash || exit $?
            cd -
        fi
        if [ -f "$1/setup" ]; then
            cd $1
            source setup || exit $?
            cd -
        fi
        # Clean
        make -C $1 clean || exit $?
        # Clean installation; ignore error if install-clean doesn't work
        # (probably because there is no install clean for most targets)
        make -C $1 install-clean || true
        # Make
        make -C $1 || exit $?
        # Install if needed
        if [ "$2" == "install" ]; then
            make -C $1 install || exit $?
        fi
    fi
}

do_make spinnaker_tools
do_make spinn_common install
do_make SpiNNFrontEndCommon/c_common/front_end_common_lib install
do_make SpiNNFrontEndCommon/c_common/ install
do_make sPyNNaker/neural_modelling/
do_make sPyNNaker8NewModelTemplate/c_models/
do_make SpiNNakerGraphFrontEnd/spinnaker_graph_front_end/examples/
do_make spinngym/c_code/
