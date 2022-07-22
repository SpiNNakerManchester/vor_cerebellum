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

from spinnaker_testbase import ScriptChecker
from unittest import SkipTest  # pylint: disable=unused-import


class TestScripts(ScriptChecker):
    """
    This file tests the scripts as configured in script_builder.py

    Please do not manually edit this file.
    It is rebuilt each time SpiNNakerManchester/IntegrationTests is run

    If it is out of date please edit and run script_builder.py
    Then the new file can be added to github for reference only.
    """
# flake8: noqa

    def test_vor_cerebellum_full_scale_tests_cerebellum_and_icub_env(self):
        self.check_script("vor_cerebellum/full_scale_tests/cerebellum_and_icub_env.py")

    def test_vor_cerebellum_full_scale_tests_vrpss_cerebellum_tb(self):
        raise SkipTest("Not sure what this is")
        self.check_script("vor_cerebellum/full_scale_tests/vrpss_cerebellum_tb.py")

    def test_vor_cerebellum_full_scale_tests_target_reaching_experiment(self):
        self.check_script("vor_cerebellum/full_scale_tests/target_reaching_experiment.py")

    def test_vor_cerebellum_full_scale_tests_vor_analysis(self):
        raise SkipTest("Just analysis, doesn't use SpiNNaker")
        self.check_script("vor_cerebellum/full_scale_tests/vor_analysis.py")

    def test_vor_cerebellum_full_scale_tests_cerebellum_experiment(self):
        self.check_script("vor_cerebellum/full_scale_tests/cerebellum_experiment.py")

    def test_vor_cerebellum_full_scale_tests_single_run_cerebellum_tb(self):
        raise SkipTest("Long - tested elsewhere")
        self.check_script("vor_cerebellum/full_scale_tests/single_run_cerebellum_tb.py")

    def test_vor_cerebellum_full_scale_tests_cerebellum_from_file(self):
        raise SkipTest("No file available to run")
        self.check_script("vor_cerebellum/full_scale_tests/cerebellum_from_file.py")
