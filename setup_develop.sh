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
