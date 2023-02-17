# Neuromorphic cerebellum learning to perform the vestibulo-ocular reflex (VOR)


This work presents the implementation of a biologically inspired and adaptive robot control system on neuromorphic hardware.
The control system is built from spiking neurons organised in a structure inspired by the cerebellum.
The network learns to perform the vestibulo-ocular reflex (VOR) in real-time.
The system is shown to perform the task in simulated and physical environments.

 This work uses a model designed by Naveros et al
 (F. Naveros, J. A. Garrido, A. Arleo, E. Ros and N. R. Luque,
 *Exploring Vestibulo-Ocular Adaptation in a Closed-Loop Neuro-Robotic Experiment Using STDP. A Simulation Study,*
 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, 2018, pp. 1-9,
 doi: [10.1109/IROS.2018.8594019](https://doi.org/10.1109/IROS.2018.8594019).).

 The model is ported onto SpiNNaker and simulated using [sPyNNaker](https://github.com/spinnakermanchester/spynnaker).
 The circuit's ability is shown in a tightly-coupled simulated environment ([SpiNNGym](https://github.com/SpiNNakerManchester/SpiNNGym)),
 in a loosely-coupled simulated environment ([NRP](https://neurorobotics.net/)) and in a physical robot ([iCub](https://icub.iit.it/)).

## Installation

 This installation assumes the existence of either sPyNNaker or [NEST](https://www.fz-juelich.de/en/inm/inm-6/leistungen/nest), for brevity.
For installation instructions for NEST, please follow official directions. For SpiNNaker, 3 shell scripts are provided
for quick installation.

A complete installation (assuming no errors) could be done by running (in this order):

```bash
./install.sh && ./setup_develop.sh && ./automatic_make.sh
```

 The software package can be installed in a local virtual environment.
The following can be run inside the unarchived folder to install the package:

```
python setup.py develop
```

or

```
pip install .
```


## Use

Each experiment script is stand-alone. However, full-scale experiments can have arguments passed to them via the
argparser.

## Citation

When citing this work, the authors would prefer you to cite the [related journal article](https://doi.org/10.3389/fncel.2021.622870).
