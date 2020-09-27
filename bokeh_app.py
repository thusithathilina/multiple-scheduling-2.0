import pickle
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, layout
from bokeh.models import ColumnDataSource, CustomJS, DataTable, TableColumn, RadioButtonGroup, DatePicker, Select
from os import walk
from scripts.input_parameter import *
from datetime import date


exp_date = "2020-09-28"
exp_time = "03-57-48-505121"


def view_results(date_folder, time_folder):

    def dict_to_pd_dt(area_res, target_dict, k0_ks, k1_ks):
        for k0 in k0_ks:
            if k0 not in target_dict:
                target_dict[k0] = dict()
            for k1 in k1_ks:
                df = pd.DataFrame.from_dict(area_res[k0][k1], orient='index')
                df.columns = [str(x) for x in range(len(area_res[k0][k1][0]))]
                target_dict[k0][k1] = df
                # target_dict[k0][k1].stack().reset_index()

    def combine_dict_to_pd_dt(area_res, target_dict, k0_ks, k1_ks):
        for k1 in k1_ks:
            combined_dict = dict()
            for k0 in k0_ks:
                combined_dict[k0] = area_res[k0][k1]
            df = pd.DataFrame(combined_dict, columns=k0_ks)
            target_dict[k1] = df

    # update data trackers
    def load_data(f_date, f_time):
        date_time_folder = "results/{}/{}/".format(f_date, f_time)
        with open(date_time_folder + "area_output.pkl", 'rb') as f2:
            area_res = pickle.load(f2)
            f2.close()

        k0_keys = [k0_demand]
        k1_keys = [k1_optimal_scheduling, k1_heuristic_scheduling]
        dict_to_pd_dt(area_res, prices_dict, k0_keys, k1_keys)

        k0_keys = [k0_demand, k0_prices]
        k1_keys = [k1_optimal_fw, k1_heuristic_fw]
        dict_to_pd_dt(area_res, demands_prices_fw_dict, k0_keys, k1_keys)

        k0_keys = [k0_demand_max, k0_par, k0_obj, k0_cost, k0_penalty, k0_step]
        if k0_time in area_res:
            k0_keys.append(k0_time)
        if k0_demand_total in area_res:
            k0_keys.append(k0_demand_total)
        k1_keys = [k1_optimal_fw, k1_heuristic_fw]
        combine_dict_to_pd_dt(area_res, others_combined_dict, k0_keys, k1_keys)

    # ---------- initialise data trackers ----------

    prices_dict = dict()
    demands_prices_fw_dict = dict()
    others_combined_dict = dict()
    load_data(date_folder, time_folder)

    # ---------- initialise web widgets ----------

    # a date picker for choosing the date of the results
    date_default = date_folder
    date_options = [dirs for root, dirs, _ in walk("results") if dirs != []][0]
    date_min = date_options[0]
    date_max = str(date.today())
    date_picker = DatePicker(title='Select date:', value=date_default, min_date=date_min, max_date=date_max)

    # a dropdown list for choosing the time of the results
    selected_date = date_picker.value
    select_options = [dirs for root, dirs, _ in walk("results/{}".format(selected_date)) if dirs != []][0]
    select = Select(title="Select time:", value=time_folder, options=select_options)

    # a dropdown list for choosing the algorithm
    selected_algorithm = k1_optimal_fw
    selected_algorithm_options = [k1_optimal_fw, k1_heuristic_fw]
    select_algorithm = Select(title="Select algorithm:", value=selected_algorithm, options=selected_algorithm_options)

    # radio buttons for choosing the date source
    LABELS = ["Demand Profile", "Price Profile", "Others"]
    radio_button_group = RadioButtonGroup(labels=LABELS, active=2)

    # a data table for displaying numbers
    source = ColumnDataSource(others_combined_dict[select_algorithm.value])
    display_keys = list(others_combined_dict[select_algorithm.value].keys())
    # source = ColumnDataSource(demands_prices_fw_dict[k0_prices][k1_optimal_scheduling_fw])
    # display_keys = demands_prices_fw_dict[k0_prices][k1_optimal_scheduling_fw].keys()
    columns = [TableColumn(field=str(i), title=str(i).replace("_", " ").capitalize()) for i in display_keys]
    data_table = DataTable(source=source, columns=columns)

    # ---------- event functions for widgets ----------

    def update_select_options(attr, old, new):
        select_opt = [dirs for root, dirs, _ in walk("results/{}".format(new)) if dirs != []][0]
        select.options = select_opt
        select.value = select_opt[-1]

    def update_data_table_content(attr, old, new):
        # print(select_algorithm.value)
        if new == 0:  # demand
            source.data = demands_prices_fw_dict[k0_demand][select_algorithm.value]
        elif new == 1:  # prices
            source.data = demands_prices_fw_dict[k0_prices][select_algorithm.value]
        elif new == 2:  # others
            # display_keys = list(source.data.keys())
            source.data = others_combined_dict[select_algorithm.value]
        # print(source.data)
        display_keys = source.data.keys()
        data_table.columns = [TableColumn(field=str(i),
                                          title=str(i).replace("_", " ").capitalize()) for i in display_keys]
        data_table.update()

    def update_data_table_source(attr, old, new):
        d_folder = date_picker.value
        t_folder = new
        load_data(d_folder, t_folder)
        active_radio_button = radio_button_group.active
        update_data_table_content(None, None, active_radio_button)

    def switch_algorithm(attr, old, new):
        active_radio_button = radio_button_group.active
        update_data_table_content(None, None, active_radio_button)

    # ---------- assign event functions to widgets ----------

    date_picker.on_change("value", update_select_options)
    select.on_change("value", update_data_table_source)
    select_algorithm.on_change("value", switch_algorithm)
    radio_button_group.on_change("active", update_data_table_content)

    # ---------- design layout ----------

    l = layout([
        [date_picker, select, select_algorithm],
        [radio_button_group],
        [data_table]

    ], sizing_mode='scale_width')

    # show graphs
    curdoc().add_root(l)


view_results(exp_date, exp_time)


