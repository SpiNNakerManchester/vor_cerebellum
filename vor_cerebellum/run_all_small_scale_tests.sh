# RUNNING ALL SMALL SCALE EXPERIMENTS IN PARALLEL

# Generic tests
nohup python generic_tests/test_neuron_params.py spinnaker > generic_tests_test_neuron_params.out 2>&1 & 
nohup python generic_tests/test_vrpss.py  > generic_tests_test_vrpss.out 2>&1 & 
nohup python generic_tests/100_spikes_rb_ls_test.py  > generic_tests_100_spikes_rb_ls_test.out 2>&1 & 
# MF-VN tests
nohup python mf_vn_tests/mf_vn_ltd_curve.py  > mf_vn_tests_mf_vn_ltd_curve.out 2>&1 & 
nohup python mf_vn_tests/single_vestibular_nuclei_mf_windowing.py  > mf_vn_tests_single_vestibular_nuclei_mf_windowing.out 2>&1 & 
nohup python mf_vn_tests/test_mfvn_lut.py  > mf_vn_tests_test_mfvn_lut.out 2>&1 & 
nohup python mf_vn_tests/single_vestibular_nuclei_potentiation_test.py  > mf_vn_tests_single_vestibular_nuclei_potentiation_test.out 2>&1 & 
nohup python mf_vn_tests/single_vestibular_nuclei_mf_windowing_3_spikes.py  > mf_vn_tests_single_vestibular_nuclei_mf_windowing_3_spikes.out 2>&1 & 

# pf-PC tests
nohup python pf_pc_tests/pf_pc_ltd_curve.py  > pf_pc_tests_pf_pc_ltd_curve.out 2>&1 & 
nohup python pf_pc_tests/single_purkinje_cell_test.py  > pf_pc_tests_single_purkinje_cell_test.out 2>&1 & 
nohup python pf_pc_tests/test_pfpc_lut.py  > pf_pc_tests_test_pfpc_lut.out 2>&1 & 
nohup python pf_pc_tests/single_purkinje_cell_potentiation_test.py  > pf_pc_tests_single_purkinje_cell_potentiation_test.out 2>&1 & 
nohup python pf_pc_tests/single_purkinje_cell_pf_windowing.py  > pf_pc_tests_single_purkinje_cell_pf_windowing.out 2>&1 & 

# spiNNGym tests
nohup python spinngym_tests/icub_vor_env_test_perfect_motion.py  > spinngym_tests_icub_vor_env_test_perfect_motion.out 2>&1 & 
nohup python spinngym_tests/icub_vor_env_test_200_inputs.py  > spinngym_tests_icub_vor_env_test_200_inputs.out 2>&1 & 
nohup python spinngym_tests/icub_vor_env_test.py  > spinngym_tests_icub_vor_env_test.out 2>&1 & 
	