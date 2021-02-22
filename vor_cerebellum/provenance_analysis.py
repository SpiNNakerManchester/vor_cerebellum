import itertools

from vor_cerebellum.utilities import *
import pandas as pd
from os.path import join as join
from numpy.polynomial.polynomial import Polynomial
from matplotlib.ticker import MultipleLocator
import traceback


def extract_per_pop_placements(df, pops):
    placement_results = {}
    for pop in pops:
        pop_df = df[df['pop'] == pop]
        placement_df = pop_df[["x", "y", "p", "no_atoms"]].drop_duplicates()
        placement_results[pop] = placement_df
    return placement_results


def extract_per_pop_info(df, type_of_prov, pops, report=False):
    pop_results = {k: None for k in pops}
    pop_results['global_mean'] = np.nan
    pop_results['global_max'] = np.nan
    pop_results['global_min'] = np.nan

    prov_filter = df['prov_name'] == type_of_prov
    filtered_prov = df[prov_filter]
    if report:
        print("{:40} for populations:".format(type_of_prov))

    _means = []
    _maxs = []
    _mins = []
    for pop in pops:
        pop_df = filtered_prov[filtered_prov['pop'] == pop]
        curr_pop_values = pop_df.prov_value
        _mean = curr_pop_values.mean()
        _max = curr_pop_values.max()
        _min = curr_pop_values.min()
        if report:
            print("\t{:25} - avg {:10.2f} max {:10.2f}".format(
                pop, curr_pop_values.mean(), curr_pop_values.max()))
        # save values
        pop_results[pop] = {
            'mean': _mean,
            'max': _max,
            'min': _min,
            'all': curr_pop_values
        }
        _means.append(_mean)
        _maxs.append(_max)
        _mins.append(_min)
    if report:
        write_line()
    pop_results['global_mean'] = np.nanmean(np.asarray(_means).astype(np.float))
    pop_results['global_max'] = np.nanmax(np.asarray(_maxs).astype(np.float))
    pop_results['global_min'] = np.nanmin(np.asarray(_mins).astype(np.float))
    return pop_results


def provenance_npz_analysis(in_file, fig_folder, run_no):
    write_header("Reading provenances in file " + in_file)
    existing_data = np.load(in_file, allow_pickle=True)
    curr_run_np = existing_data[str(run_no)]

    prov = pd.DataFrame.from_records(curr_run_np)
    pops = prov['pop'].unique()
    pops.sort()
    types_of_provenance = prov['prov_name'].unique()
    prov_of_interest = [
        'MAX_SPIKES_IN_A_TICK',
        'Times_synaptic_weights_have_saturated',
        'late_packets',
        'Times_the_input_buffer_lost_packets',
        'Times_the_timer_tic_over_ran',
        'Total_pre_synaptic_events',
        'MAX_DMAS_IN_A_TICK',
        'MAX_PIPELINE_RESTARTS',
        'send_multicast_packets',
        'MAX_FLUSHED_SPIKES',
        'TOTAL_FLUSHED_SPIKES'
    ]

    results = {k: None for k in types_of_provenance}
    # TODO report number of neurons to make sure the networks is correct
    write_short_msg("DETECTED POPULATIONS", pops)

    for type_of_prov in types_of_provenance:
        rep = True if type_of_prov in prov_of_interest else False
        results[type_of_prov] = extract_per_pop_info(prov, type_of_prov, pops,
                                                     report=rep)
    placements = extract_per_pop_placements(prov, pops)
    return results, types_of_provenance, prov_of_interest, placements


def provenance_analysis(in_file, fig_folder):
    # Check if the folders exist
    if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
        os.mkdir(fig_folder)
    current_fig_folder = join(fig_folder, in_file.split('/')[-1])
    # Make folder for current figures
    if not os.path.isdir(current_fig_folder) and not os.path.exists(current_fig_folder):
        os.mkdir(current_fig_folder)

    write_header("Analysing provenance in archive:" + in_file)
    # read the file
    existing_data = np.load(in_file, allow_pickle=True)
    # TODO REPORT metadata

    # figure out the past run id
    numerical_runs = [int(x) for x in existing_data.files
                      if x not in ["metadata"]]
    numerical_runs.sort()
    collated_results = {k: None for k in numerical_runs}
    types_of_provenance = None
    prov_of_interest = None
    placements = {}

    router_provenance_of_interest = [
        'Dumped_from_a_Link',
        'Dumped_from_a_processor',
        'Local_Multicast_Packets',
        'External_Multicast_Packets',
        'Dropped_Multicast_Packets',
        'Missed_For_Reinjection'
    ]
    router_pop_names = ['router_provenance']

    write_short_msg("Number of runs", len(numerical_runs))
    write_sep()
    for run_no in numerical_runs:
        collated_results[run_no], types_of_provenance, \
        prov_of_interest, placements[run_no] = provenance_npz_analysis(
            in_file, fig_folder, run_no)

    # if group_on_name is None:
    write_header("REPORTING BEST SIMULATIONS")
    cumulative_report(collated_results, types_of_provenance,
                      prov_of_interest)

    plot_per_population_provenance_of_interest(
        collated_results,
        router_provenance_of_interest,
        current_fig_folder,
        router_pop=router_pop_names)

    plot_population_placement(collated_results, placements,
                              fig_folder=current_fig_folder)

    for run_no in numerical_runs:
        plot_router_provenance(in_file, run_no, router_pop_names,
                               router_provenance_of_interest,
                               current_fig_folder)

    plot_per_population_provenance_of_interest(collated_results,
                                               prov_of_interest,
                                               current_fig_folder)

    for run_no in numerical_runs:
        plot_2D_map_for_poi(in_file, run_no,
                            prov_of_interest, router_pop_names,
                            current_fig_folder, placements)


def plot_2D_map_for_poi(in_file, selected_sim,
                        provenance_of_interest, router_pop_names, fig_folder, placements):
    write_header("PLOTTING MAPS FOR ALL PRROVENANCE OF INTEREST")

    existing_data = np.load(in_file, allow_pickle=True)
    curr_run_np = existing_data[str(selected_sim)]

    prov = pd.DataFrame.from_records(curr_run_np)
    # Filter out router provenance because the logic for plotting those maps is slightly different
    pop_only_prov = prov[~prov['pop'].isin(router_pop_names)]
    filtered_placement = \
        placements[selected_sim]
    try:
        router_provenance = filtered_placement['router_provenance']
    except KeyError:
        traceback.print_exc()
        router_provenance = filtered_placement

    for type_of_provenance in provenance_of_interest:
        #  need to get processor p as well as x y prov_value
        filtered_placement = \
            pop_only_prov[pop_only_prov.prov_name == type_of_provenance][['x', 'y', 'p', 'prov_value']]
        if filtered_placement.shape[0] == 0:
            write_short_msg("NO INFORMATION FOR PROVENANCE", type_of_provenance)
            continue

        # make a new directory for each provenance
        # Check if the results folder exist
        per_prov_dir = os.path.join(fig_folder, type_of_provenance.lower())
        if not os.path.isdir(per_prov_dir) and not os.path.exists(per_prov_dir):
            os.mkdir(per_prov_dir)
        # Plotting bit
        # Fake printing to start things off...
        f = plt.figure(1, figsize=(9, 9), dpi=400)
        plt.close(f)
        # Compute plot order
        plot_order = get_plot_order(router_pop_names)
        plot_display_names = []
        for po in plot_order:
            plot_display_names.append(use_display_name(po))

        magic_constant = 4
        max_x = (router_provenance.x.max() + 1) * magic_constant
        max_y = (router_provenance.y.max() + 1) * magic_constant

        x_ticks = np.arange(0, max_x, magic_constant)[::2]
        x_tick_lables = (x_ticks / magic_constant).astype(int)
        y_ticks = np.arange(0, max_y, magic_constant)[::2]
        y_tick_lables = (y_ticks / magic_constant).astype(int)
        map = np.ones((max_x, max_y)) * np.nan

        for row_index, row in filtered_placement.iterrows():
            x_pos = int(magic_constant * row.x +
                        ((row.p // magic_constant) % magic_constant))
            y_pos = int(magic_constant * row.y +
                        (row.p % magic_constant))
            map[y_pos, x_pos] = row.prov_value

        # crop_point = np.max(np.max(np.argwhere(np.isfinite(map)), axis=0))
        f = plt.figure(1, figsize=(9, 9), dpi=500)
        # plt.matshow(map[:crop_point, :crop_point], interpolation='none')
        im = plt.imshow(map, interpolation='none',
                        cmap=plt.get_cmap('inferno'),
                        extent=[0, max_x, 0, max_y],
                        origin='lower')
        ax = plt.gca()

        plt.xlabel("Chip X coordinate")
        plt.ylabel("Chip Y coordinate")

        plt.xticks(x_ticks, x_tick_lables)
        plt.yticks(y_ticks, y_tick_lables)
        ax.yaxis.set_minor_locator(MultipleLocator(magic_constant))
        ax.xaxis.set_minor_locator(MultipleLocator(magic_constant))

        plt.grid(b=True, which='both', color='k', linestyle='-')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(use_display_name(type_of_provenance))

        save_figure(plt, join(per_prov_dir,
                              "map_of_{}_for_{}".format(type_of_provenance,
                                                        selected_sim)),
                    extensions=['.png', '.pdf'])
        plt.close(f)


def plot_router_provenance(in_file, selected_sim, router_pop_names,
                           router_provenance_of_interest, fig_folder):
    write_header("PLOTTING ROUTER INFO AND MAPS")
    existing_data = np.load(in_file, allow_pickle=True)
    curr_run_np = existing_data[str(selected_sim)]

    prov = pd.DataFrame.from_records(curr_run_np)
    # prov = pd.read_csv(join(join(folder, selected_sim), "structured_provenance.csv"))
    # Need to filter only info for routers
    # then filter by type of router provenance
    # extract X, Y, prov_value
    # Plot map
    router_only_prov = prov[prov['pop'].isin(router_pop_names)]

    for type_of_provenance in router_provenance_of_interest:
        filtered_placement = \
            router_only_prov[router_only_prov.prov_name == type_of_provenance][['x', 'y', 'prov_value']]
        if filtered_placement.shape[0] == 0:
            write_short_msg("NO INFORMATION FOR PROVENANCE", type_of_provenance)
            continue

        # make a new directory for each provenance
        # Check if the results folder exist
        per_prov_dir = os.path.join(fig_folder, type_of_provenance.lower())
        if not os.path.isdir(per_prov_dir) and not os.path.exists(per_prov_dir):
            os.mkdir(per_prov_dir)
        # Plotting bit
        # Fake printing to start things off...
        f = plt.figure(1, figsize=(9, 9), dpi=400)
        plt.close(f)
        # Compute plot order
        plot_order = get_plot_order(router_pop_names)
        plot_display_names = []
        for po in plot_order:
            plot_display_names.append(use_display_name(po))

        magic_constant = 4
        max_x = (filtered_placement.x.max() + 1) * magic_constant
        max_y = (filtered_placement.y.max() + 1) * magic_constant
        x_ticks = np.arange(0, max_x, magic_constant)[::2]
        # x_tick_lables = np.linspace(0, collated_placements.x.max(), 6).astype(int)
        x_tick_lables = (x_ticks / magic_constant).astype(int)
        y_ticks = np.arange(0, max_y, magic_constant)[::2]
        # y_tick_lables = np.linspace(0, collated_placements.y.max(), 6).astype(int)
        y_tick_lables = (y_ticks / magic_constant).astype(int)
        map = np.ones((max_x, max_y)) * np.nan

        for row_index, row in filtered_placement.iterrows():
            map[
            int(magic_constant * row.y):int(magic_constant * (row.y + 1)),
            int(magic_constant * row.x):int(magic_constant * (row.x + 1))
            ] = row.prov_value

        f = plt.figure(1, figsize=(9, 9), dpi=500)
        im = plt.imshow(map, interpolation='none',
                        cmap=plt.get_cmap('inferno'),
                        extent=[0, max_x, 0, max_y],
                        origin='lower')
        ax = plt.gca()

        plt.xlabel("Chip X coordinate")
        plt.ylabel("Chip Y coordinate")

        plt.xticks(x_ticks, x_tick_lables)
        plt.yticks(y_ticks, y_tick_lables)
        ax.yaxis.set_minor_locator(MultipleLocator(magic_constant))
        ax.xaxis.set_minor_locator(MultipleLocator(magic_constant))

        plt.grid(b=True, which='both', color='k', linestyle='-')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(use_display_name(type_of_provenance))

        save_figure(plt, join(per_prov_dir,
                              "map_of_{}_for_{}".format(type_of_provenance,
                                                        selected_sim)),
                    extensions=['.png', '.pdf'])
        plt.close(f)


def cumulative_report(collated_results, types_of_provenance, prov_of_interest):
    sorted_key_list = list(collated_results.keys())
    sorted_key_list.sort()
    for type_of_prov in types_of_provenance:
        of_interest = True if type_of_prov in prov_of_interest else False
        if not of_interest:
            continue
        # Need to loop over all dicts in collated_results
        _means = []
        _maxs = []
        _mins = []
        _keys = []
        _max_values_per_pop = None

        print("{:40} for all cases".format(type_of_prov))
        for k in sorted_key_list:
            v = collated_results[k]
            filtered_v = v[type_of_prov]
            _keys.append(k)
            _means.append(filtered_v['global_mean'])
            _maxs.append(filtered_v['global_max'])
            _mins.append(filtered_v['global_min'])
            print("run {:40} | min {:10.2f} | mean {:10.2f} | max {:10.2f}".format(
                k, filtered_v['global_min'], filtered_v['global_mean'], filtered_v['global_max']
            ))
            if _max_values_per_pop is None:
                _max_values_per_pop = {x: [] for x in filtered_v.keys() if "cell" in x}
            for vp in _max_values_per_pop.keys():
                _max_values_per_pop[vp].append(filtered_v[vp]['max'])
        # Report cumulative stats per population
        write_line()
        print("{:40} for all populations".format(type_of_prov))
        reporting_keys = list(_max_values_per_pop.keys())
        reporting_keys.sort()
        for rk in reporting_keys:
            vals = _max_values_per_pop[rk]
            print("{:40} | mean {:10.2f} | max {:10.2f} | std {:10.2f}".format(
                rk, np.nanmean(vals), np.nanmax(vals), np.nanstd(vals)
            ))


def plot_population_placement(collated_results, placements, fig_folder):
    write_header("PLOTTING MAPS")
    sorted_key_list = list(collated_results.keys())
    sorted_key_list.sort()
    for selected_sim in sorted_key_list:
        filtered_placement = \
            placements[selected_sim]
        # try:
        router_provenance = filtered_placement['router_provenance']

        placements_per_pop = {x: filtered_placement[x]
                              for x in filtered_placement.keys()
                              if 'router_provenance' not in x}
        # make a new directory for each provenance
        # Check if the results folder exist
        per_prov_dir = os.path.join(fig_folder, "placements")
        if not os.path.isdir(per_prov_dir) and not os.path.exists(per_prov_dir):
            os.mkdir(per_prov_dir)

        # Plotting bit
        # Fake printing to start things off...
        f = plt.figure(1, figsize=(9, 9), dpi=400)
        plt.close(f)
        # Compute plot order
        plot_order = get_plot_order(placements_per_pop.keys())
        plot_display_names = []
        for po in plot_order:
            plot_display_names.append(use_display_name(po))
        n_plots = len(plot_order)

        collated_placements = pd.concat([
            filtered_placement[x] for x in plot_order
        ])

        magic_constant = 4

        max_x = (router_provenance.x.max() + 1) * magic_constant
        max_y = (router_provenance.y.max() + 1) * magic_constant

        x_ticks = np.arange(0, max_x, magic_constant)[::2]
        x_tick_lables = (x_ticks / magic_constant).astype(int)
        y_ticks = np.arange(0, max_y, magic_constant)[::2]
        y_tick_lables = (y_ticks / magic_constant).astype(int)
        map = np.ones((max_x, max_y)) * np.nan
        for index, pop in enumerate(plot_order):
            curr_pl = placements_per_pop[pop]
            for row_index, row in curr_pl.iterrows():
                x_pos = int(magic_constant * row.x +
                            ((row.p // magic_constant) % magic_constant))
                y_pos = int(magic_constant * row.y +
                            (row.p % magic_constant))
                map[y_pos, x_pos] = index

        uniques = np.unique(map[np.isfinite(map)]).astype(int)
        f = plt.figure(1, figsize=(9, 9), dpi=500)
        im = plt.imshow(map, interpolation='none', vmin=0, vmax=n_plots,
                        cmap=plt.get_cmap('viridis', n_plots),
                        extent=[0, max_x, 0, max_y],
                        origin='lower')
        ax = plt.gca()

        plt.xlabel("Chip X coordinate")
        plt.ylabel("Chip Y coordinate")

        plt.xticks(x_ticks, x_tick_lables)
        plt.yticks(y_ticks, y_tick_lables)
        ax.yaxis.set_minor_locator(MultipleLocator(magic_constant))
        ax.xaxis.set_minor_locator(MultipleLocator(magic_constant))

        plt.grid(b=True, which='both', color='k', linestyle='-')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Population")
        cbar.ax.set_yticks(uniques)
        cbar.ax.set_yticklabels(plot_display_names)

        save_figure(plt, join(per_prov_dir,
                              "map_of_placements_for_run_{}".format(selected_sim)),
                    extensions=['.png', '.pdf'])
        plt.close(f)

        # Some reports
        write_short_msg("Plotting map for", selected_sim)
        write_short_msg("Number of cores used", collated_placements.shape[0])
        write_short_msg("Number of chips used",
                        collated_placements[["x", "y"]].drop_duplicates().shape[0])
        write_short_msg("Unique pop ids", uniques)
        write_line()


def plot_per_population_provenance_of_interest(
        collated_results, prov,
        fig_folder, router_pop=None):
    if router_pop is not None:
        write_header("PLOTTING PER ROUTER VALUES FOR PROVENANCE")
    else:
        write_header("PLOTTING PER POPULATION VALUES FOR PROVENANCE")
    sorted_key_list = list(collated_results.keys())
    sorted_key_list.sort()
    curr_poi = sorted_key_list
    f = plt.figure(1, figsize=(9, 9), dpi=400)
    plt.close(f)
    curr_group = "Runs"
    for type_of_prov in prov:
        curr_mapping = {val: None for val in sorted_key_list}
        _max_values_per_pop = {}
        for curr_run in sorted_key_list:
            # If Provenance type is not present (maybe looking for a provenance type for LIF neurons not SSAs)
            if type_of_prov not in collated_results[curr_run].keys() and router_pop is not None:
                filtered_collated_results = {x: None for x in router_pop}
                for rp in router_pop:
                    filtered_collated_results[rp] = {
                        'max': np.nan,
                        'all': pd.DataFrame([np.nan])}
            elif type_of_prov in collated_results[curr_run].keys():
                filtered_collated_results = \
                    collated_results[curr_run][type_of_prov]
            elif type_of_prov in collated_results[curr_run].keys() and router_pop is not None:
                continue
            if router_pop is not None:
                _max_values_per_pop = \
                    {x: None for x in filtered_collated_results.keys()
                     if x in router_pop}
            else:
                _max_values_per_pop = \
                    {x: None for x in filtered_collated_results.keys()
                     if ("router" not in x and
                         hasattr(filtered_collated_results[x], "__len__"))}
            for vp in _max_values_per_pop.keys():

                _max_values_per_pop[vp] = [list(filtered_collated_results[vp]['all'].values)]

            # Need to create a list of entries per pop
            if curr_mapping[curr_run] is None:
                curr_mapping[curr_run] = _max_values_per_pop
            else:
                for k, v in _max_values_per_pop.items():
                    if v:
                        curr_mapping[curr_run][k].extend(v)
        # Plotting bit
        plot_order = get_plot_order(_max_values_per_pop.keys())
        n_plots = float(len(plot_order))

        # Convert all 2D arrays of results to numpy arrays
        # Can report intra-trial and inter-trial variability here
        for k in curr_mapping.keys():
            if curr_mapping[k]:
                for p in curr_mapping[k].keys():
                    curr_mapping[k][p] = np.array(curr_mapping[k][p])

        f = plt.figure(1, figsize=(9, 9), dpi=400)
        for index, pop in enumerate(plot_order):
            curr_median = []
            curr_percentiles = []
            for k in curr_poi:
                if curr_mapping[k] and curr_mapping[k][pop].size > 0:
                    merged = np.array(
                        list(itertools.chain.from_iterable(curr_mapping[k][pop]))).astype(np.float)
                    curr_median.append(np.nanmedian(merged))
                    curr_percentiles.append([np.nanmedian(merged) - np.percentile(merged, 5),
                                             np.percentile(merged, 95) - np.nanmedian(merged)])
                else:
                    curr_median.append(np.nan)
                    curr_percentiles.append(np.nan)

            curr_median = np.array(curr_median)
            curr_percentiles = np.array(curr_percentiles).T
            if np.any(np.isfinite(curr_median)):
                plt.errorbar(curr_poi, curr_median,
                             yerr=curr_percentiles,
                             color=color_for_index(index, n_plots),
                             marker='o',
                             label=use_display_name(pop),
                             alpha=0.8)
                # also print out the values per pop to easily copy and paste
                write_short_msg(use_display_name(pop), curr_median)
                write_short_msg(use_display_name(pop), curr_percentiles)
                try:
                    # TODO also do some curve fitting for these numbers
                    for deg in [1, 2]:
                        fit_res = polyfit(curr_poi, curr_median, deg)
                        write_short_msg("degree poly {} coeff of "
                                        "determination".format(deg),
                                        fit_res['determination'])
                    fit_res = polyfit(curr_poi, np.log(curr_median), 1)
                    write_short_msg("exp fit coeff of "
                                    "determination",
                                    fit_res['determination'])
                except:
                    traceback.print_exc()

        plt.xlabel(use_display_name(curr_group))
        plt.ylabel(use_display_name(type_of_prov))
        ax = plt.gca()
        plt.legend(loc='best')
        plt.tight_layout()
        save_figure(plt, join(fig_folder, "median_{}".format(type_of_prov)),
                    extensions=['.png', '.pdf'])
        if "MAX" in type_of_prov:
            ax.set_yscale('log')
            save_figure(plt, join(fig_folder, "log_y_median_{}".format(type_of_prov)),
                        extensions=['.png', '.pdf'])
        plt.close(f)

        f = plt.figure(1, figsize=(9, 9), dpi=400)
        for index, pop in enumerate(plot_order):
            curr_median = []
            curr_percentiles = []
            for k in curr_poi:
                if curr_mapping[k] and curr_mapping[k][pop].size > 0:
                    merged = np.array(
                        list(itertools.chain.from_iterable(curr_mapping[k][pop]))).astype(np.float)
                    curr_median.append(np.nanmean(merged))
                    curr_percentiles.append(np.nanstd(merged))
                else:
                    curr_median.append(np.nan)
                    curr_percentiles.append(np.nan)
            curr_median = np.array(curr_median)
            if np.any(np.isfinite(curr_median)):
                plt.errorbar(curr_poi, curr_median,
                             yerr=curr_percentiles,
                             color=color_for_index(index, n_plots),
                             marker='o',
                             label=use_display_name(pop),
                             alpha=0.8)
                # also print out the values per pop to easily copy and past
                # also print out the values per pop to easily copy and paste
                write_short_msg(use_display_name(pop), curr_median)
                write_short_msg(use_display_name(pop), curr_percentiles)
                # TODO also do some curve fitting for these numbers
                try:
                    for deg in [1, 2]:
                        fit_res = polyfit(curr_poi, curr_median, deg)
                        write_short_msg("degree poly {} coeff of "
                                        "determination".format(deg),
                                        fit_res['determination'])
                    fit_res = polyfit(curr_poi, np.log(curr_median), 1)
                except:
                    traceback.print_exc()
                write_short_msg("exp fit coeff of "
                                "determination",
                                fit_res['determination'])

        plt.xlabel(use_display_name(curr_group))
        plt.ylabel(use_display_name(type_of_prov))
        ax = plt.gca()
        plt.legend(loc='best')
        plt.tight_layout()
        save_figure(plt, join(fig_folder, "{}".format(type_of_prov)),
                    extensions=['.png', '.pdf'])

        plt.close(f)
        if "MAX" in type_of_prov:
            ax.set_yscale('log')
            save_figure(plt, join(fig_folder, "log_y_median_{}".format(type_of_prov)),
                        extensions=['.png', '.pdf'])
        plt.close(f)


# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)  # or [p(z) for z in x]
    ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


if __name__ == "__main__":
    import sys
    input_filename = sys.argv[-1]
    provenance_analysis(input_filename, "figures/provenance_figures")

