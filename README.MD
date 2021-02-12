# Neuromorphic cerebellum learning to perform the vestibulo-ocular reflex (VOR)


This work presents the implementation of a biologically inspired and adaptive robot control system on neuromorphic hardware. 
The control system is built from spiking neurons organised in a structure inspired by the cerebellum. 
The network learns to perform the vestibulo-ocular reflex (VOR) in real-time. 
The system is shown to perform the task in simulated and physical environments.

 This work uses a model designed by Naveros et al 
 (F. Naveros, J. A. Garrido, A. Arleo, E. Ros and N. R. Luque, 
 "Exploring Vestibulo-Ocular Adaptation in a Closed-Loop Neuro-Robotic Experiment Using STDP. A Simulation Study," 
 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Madrid, 2018, pp. 1-9, 
 doi: 10.1109/IROS.2018.8594019.). 
 The model is ported onto SpiNNaker and simulated using [sPyNNaker](https://github.com/spinnakermanchester/spynnaker). 
 The circuit's ability is shown in a tightly-coupled simulated environment (SpiNNGym), 
 in a loosely-coupled simulated environment (NRP) and in a physical robot (iCub).
 
 # Installation
 
 The software package can be installed in a local virtual environment. 
The following can be run inside the unarchived folder to install the package:

```
python setup.py develop
```

or

```
pip install .
```

This installation assumes the existence of either sPyNNaker or NEST, for brevity. 
For installation instructions for those, please follow the official directions provided by each package.

# Use

Watch this space...