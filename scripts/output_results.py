from shutil import copy
import pickle
import pandas as pd
from scripts.input_parameter import *
from scripts.cfunctions import average
from pathlib import Path
import pandas_bokeh
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.models import ColumnDataSource, NumeralTickFormatter


def write_batch_experiment_summary(exp_summary_dt, group_by_cols, dt_folder, time):

    def draw_graph(df, key_selected):

        plots = []
        result_type_name = key_selected.replace("_", " ").capitalize()
        for alg in set(sorted(df[k0_algorithm].values)):
            alg_name = alg
            alg_type = "scheduling"
            if "heuristic" in alg:
                alg_name = "OGSA"
            elif "optimal" in alg:
                alg_name = "Optimisation model"
            if "fw" in alg:
                alg_name += " with FW"
                alg_type = "pricing"

            df_selected = df.loc[lambda df2: df[k0_algorithm] == alg, lambda df2: [k0_households_no, key_selected]]
            df_selected = df_selected.set_index(k0_households_no)
            y_label = ""
            if "time" in key_selected:
                y_label = "Run time (seconds)"
            elif "iteration" in key_selected:
                y_label = "Iteration"
            elif "reduction" in key_selected:
                y_label = key_selected.replace("_", " ").capitalize()
            p_line = df_selected.plot_bokeh(
                kind="line",
                title="{} of {}, {}" .format(result_type_name, alg_type, alg_name),
                xlabel="Number of Households",
                ylabel=y_label,
                show_figure=False,
                sizing_mode="scale_width",
            )
            if "reduction" in key_selected:
                p_line.yaxis.formatter = NumeralTickFormatter(format='0 %')
            p_line.legend.location = "top_left"
            plots.append(p_line)

        pandas_bokeh.output_file("{}{}.html".format(dt_folder, result_type_name))
        grid = pandas_bokeh.plot_grid(plots, ncols=2, show_plot=False)
        pandas_bokeh.save(grid)

    experiment_summary_pd = pd.DataFrame.from_dict(exp_summary_dt, orient='index')
    experiment_summary_pd.index.names = ["repeat", "drop", k0_algorithm]
    experiment_summary_pd = experiment_summary_pd.reset_index()
    experiment_summary_pd = experiment_summary_pd.drop(columns=["drop"])
    experiment_summary_pd.to_csv(dt_folder + "{}_summary.csv".format(time))

    experiment_overview_pd = experiment_summary_pd.groupby(group_by_cols).mean()
    experiment_overview_pd.to_csv(dt_folder + "{}_overview.csv".format(time))

    df_runtime = experiment_overview_pd.reset_index()[[k0_households_no, k0_algorithm, k1_time_average]]
    draw_graph(df_runtime, k1_time_average)
    df_iterations = experiment_overview_pd.reset_index()[[k0_households_no, k0_algorithm, k0_iteration_no]]
    draw_graph(df_iterations, k0_iteration_no)
    df_iterations = experiment_overview_pd.reset_index()[[k0_households_no, k0_algorithm, k0_demand_max+ " reduction"]]
    draw_graph(df_iterations, k0_demand_max+ " reduction")
    df_iterations = experiment_overview_pd.reset_index()[[k0_households_no, k0_algorithm, k0_par+ " reduction"]]
    draw_graph(df_iterations, k0_par+ " reduction")
    df_iterations = experiment_overview_pd.reset_index()[[k0_households_no, k0_algorithm, k0_cost+ " reduction"]]
    draw_graph(df_iterations, k0_cost+ " reduction")


def aggregate_results(area_dt, key_params):

    def reduction(k0):
        for alg2 in area_dt:
            init_value = area_dt[alg2][k0][0]
            final_value = list(area_dt[alg2][k0].values())[-1]
            alg_reduction = (init_value - final_value) / init_value
            summary[alg2][k0 + " initial"] = init_value
            summary[alg2][k0 + " final"] = final_value
            summary[alg2][k0 + " reduction"] = round(alg_reduction, 3)

    summary = dict()
    for alg in area_dt:
        all_run_times = list(area_dt[alg][k0_time].values())[1:]
        summary[alg] = dict()
        summary[alg][k1_time_average] = round(average(all_run_times), 3)
        summary[alg][k0_iteration_no] = len(all_run_times)
        for param_k, param_v in key_params.items():
            summary[alg][param_k] = param_v

    reduction(k0_demand_max)
    reduction(k0_par)
    reduction(k0_obj)
    reduction(k0_cost)
    reduction(k0_par)

    return area_dt, summary


def draw_graphs(df_summary, df_demands_dt, df_prices_dt, df_others_dt, exp_folder):
    df_summary.index.names = [k0_algorithm]
    df_summary = df_summary.reset_index()
    data_table_summary = DataTable(
        columns=[TableColumn(field=Ci, title=Ci) for Ci in df_summary.columns],
        source=ColumnDataSource(df_summary),
        height=50,
        sizing_mode="scale_both",
        # height_policy='auto',
    )

    def place_legend(pl):
        legend = pl.legend[0]
        legend.items[0].label['value'] = "Preferred"
        legend.items[-1].label['value'] = "Optimised"
        for item in legend.items[1:-1]:
            item.label['value'] = "Itr {}".format(item.label['value'])
        legend.click_policy = "mute"
        pl.add_layout(legend, 'right')
        return pl

    rows = [[data_table_summary]]
    x_ticks = [i for i in range(48) if i % 4 == 0]
    p_height = 250
    for alg in df_demands_dt:
        alg_name = alg
        if "heuristic" in alg:
            alg_name = "OGSA and FW"
        elif "optimal" in alg:
            alg_name = "Optimisation model and FW"
        if 'fw' in alg:
            df_cost = df_others_dt[alg].T[k0_cost].div(100)
            p_line_cost = df_cost.plot_bokeh(
                kind="line",
                title="Costs, {}".format(alg_name),
                xlabel="Iteration",
                ylabel="Costs",
                show_figure=False,
                marker="circle",
            )
            p_line_cost.plot_height = p_height

            p_line_demand = df_demands_dt[alg].T.plot_bokeh(
                kind="line",
                title="Demand Profiles, {}".format(alg_name),
                xlabel="Time Period",
                ylabel="Demand (kWh)",
                xticks=x_ticks,
                show_figure=False,
                muted_alpha=0,
                muted=True,
                sizing_mode="scale_width",
                toolbar_location="above",
            )
            p_line_demand.plot_height = p_height
            p_line_demand = place_legend(p_line_demand)
            p_line_demand.renderers[0].muted = False
            p_line_demand.renderers[-1].muted = False

            p_line_price = df_prices_dt[alg].T.plot_bokeh(
                kind="line",
                title="Prices, {}".format(alg_name),
                xlabel="Time Period",
                ylabel="Price (Dollar)",
                xticks=x_ticks,
                show_figure=False,
                muted_alpha=0,
                muted=True,
                sizing_mode="scale_width",
                toolbar_location="above",
            )
            p_line_price.plot_height = p_height
            p_line_price = place_legend(p_line_price)
            p_line_price.renderers[0].muted = False
            p_line_price.renderers[-1].muted = False

            rows.append([p_line_demand, p_line_price])

    pandas_bokeh.output_file("{}plots.html".format(exp_folder))
    layout = pandas_bokeh.layout(rows, sizing_mode='scale_width')
    pandas_bokeh.save(layout)


def write_results(key_parameters, area_res, exp_folder, note):

    area_res, sum_dict = aggregate_results(area_res, key_parameters)

    path_exp_folder = Path(exp_folder)
    if not path_exp_folder.exists():
        path_exp_folder.mkdir(mode=0o777, parents=True, exist_ok=False)

    with open(exp_folder + "note" + '.txt', 'w+') as f:
        f.write(note)
    f.close()

    with open(exp_folder + "area_output" + '.pkl', 'wb+') as f:
        pickle.dump(area_res, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    with open(exp_folder + "summary" + '.pkl', 'wb+') as f:
        pickle.dump(sum_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # copy the generated data
    copy('data/area.pkl', exp_folder + "area_input.pkl")
    copy('data/households.pkl', exp_folder + "households_input.pkl")

    def save_to_csv(data_dict, file_name):
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        if k0_demand in file_name:
            df = df.div(1000)
        elif k0_prices in file_name:
            df = df.div(100)
        df.to_csv(exp_folder + "{}.csv".format(file_name))
        return df

    df_summary = save_to_csv(sum_dict, "summary")
    df_demands_dict = dict()
    df_prices_dict = dict()
    df_others_dict = dict()
    for alg in area_res:
        others_keys = {k0_demand_max, k0_demand_total, k0_par, k0_obj, k0_cost, k0_penalty, k0_step, k0_time}
        others_dict = {k: area_res[alg][k] for k in area_res[alg].keys() & others_keys}

        df_others_dict[alg] = save_to_csv(others_dict, "{}-{}".format(alg, "others"))
        df_demands_dict[alg] = save_to_csv(area_res[alg][k0_demand], "{}-{}".format(alg, k0_demand))
        df_prices_dict[alg] = save_to_csv(area_res[alg][k0_prices], "{}-{}".format(alg, k0_prices))

    draw_graphs(df_summary, df_demands_dict, df_prices_dict, df_others_dict, exp_folder)

    return sum_dict



