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

from spinnaker_testbase import RootScriptBuilder


class ScriptBuilder(RootScriptBuilder):
    """
    This file will recreate the test_scripts.py file

    To skip the too_long scripts run this script with a parameter
    """

    def build_scripts(self):
        # These scripts raise a SkipTest with the reasons given
        exceptions = {}
        exceptions["cerebellum_from_file.py"] = "No file available to run"
        exceptions["vor_analysis.py"] = "Just analysis, doesn't use SpiNNaker"
        exceptions["vrpss_cerebellum_tb.py"] = "Not sure what this is"
        exceptions["single_run_cerebellum_tb.py"] = "Long - tested elsewhere"

        # For branches these raise a SkipTest quoting the time given
        # For cron and manual runs these just add a warning
        too_long = {}

        self.create_test_scripts(["vor_cerebellum_examples/full_scale_tests"],
                                 too_long, exceptions)


if __name__ == '__main__':
    builder = ScriptBuilder()
    builder.build_scripts()
