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
	