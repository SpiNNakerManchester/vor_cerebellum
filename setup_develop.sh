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

# This script assumes it is run from the directory holding all github projects in parellel
# sh SupportScripts/settip.sh a_branch_name

do_setup(){
	cd $1 || exit $?
	python setup.py develop || exit $1
    cd ..
}

do_setup SpiNNUtils
do_setup SpiNNMachine
do_setup SpiNNStorageHandlers
do_setup SpiNNMan
do_setup PACMAN
do_setup DataSpecification
do_setup spalloc
do_setup SpiNNFrontEndCommon
do_setup sPyNNaker
cd sPyNNaker8 && python setup.py develop
cd ..
cd spinngym && python setup.py develop && cd -
