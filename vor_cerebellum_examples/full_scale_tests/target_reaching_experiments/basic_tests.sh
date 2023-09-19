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

suffix="all_target_reaching"

allSimtimes=(50000)

allGains=(80)

allSlows=(5 10)

allPOIs=("zero" "constant" "bistate" "tristate")
for simtime in ${allSimtimes[@]};do
	for gain in ${allGains[@]};do
		for slowdown in ${allSlows[@]};do
			for poi in ${allPOIs[@]};do
			name="$simtime"_gain_"$gain"_slowdown_"$slowdown"_"$poi"
			echo $suffix $gain - $simtime $slowdown "$name"_"$suffix"
			nohup python ../target_reaching_experiment.py --suffix $suffix -o $name --gain $gain --f_base 1 --f_peak 10 --simtime $simtime\
			--experiment $poi --slowdown_factor $slowdown > "$name"_"$suffix".out 2>&1 &
			done
		done
	done
done
