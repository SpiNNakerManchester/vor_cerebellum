# PF-PC plasticity testing

These scripts test the implementation of the plasticity at PF-PC synapses gated by CF activity.

The tests are:
  - [test_pfpc_lut](test_pfpc_lut.py) - testing whether the LTD Look-up Table (LUT) is computed correctly
  (are correctly taken into acccount within the plasticity window)
  - [single_purkinje_cell_potentiation_test](single_purkinje_cell_potentiation_test.py) - testing the constant LTP applied to pf-PC synapses, that is, 
  those coming from GrC, whenever 
  there is a spike from MF
  - [single_purkinje_cell_pf_windowing](single_purkinje_cell_pf_windowing.py) - testing whether spikes are correctly considered for LTD 
  (are correctly taken into acccount within the plasticity window)
  - [pf_pc_ltd_curve](pf_pc_ltd_curve.py) - testing whether the LTD curve is correct

![](figures/pf_pc_ltd_curve.png)