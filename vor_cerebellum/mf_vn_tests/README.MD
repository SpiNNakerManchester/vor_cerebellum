# MF-VN plasticity testing

These scripts test the implementation of the plasticity at MF-VN synapses gated by PC activity.

The tests are:
  - [test_mfvn_lut](test_mfvn_lut.py) - testing whether the LTD Look-up Table (LUT) is computed correctly
  (are correctly taken into acccount within the plasticity window)
  - [single_vestibular_nuclei_potentiation_test](single_vestibular_nuclei_potentiation_test.py) - testing the constant LTP applied to MF-VN synapses whenever 
  there is a spike from MF
  - [single_vestibular_nuclei_mf_windowing_3_spikes](single_vestibular_nuclei_mf_windowing_3_spikes.py) - testing whether spikes are correctly considered for LTD 
  (are correctly taken into acccount within the plasticity window) using only 3 spikes
  - [single_vestibular_nuclei_mf_windowing](single_vestibular_nuclei_mf_windowing_.py) - testing whether spikes are correctly considered for LTD 
  (are correctly taken into acccount within the plasticity window)
  - [mf_vn_ltd_curve](mf_vn_ltd_curve.py) - testing whether the LTD curve is correct
![](figures/mf_vn_ltd_curve.png)