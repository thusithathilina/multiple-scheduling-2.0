from multiple.scripts.input_parameter import *
from os import path, mkdir, walk
from datetime import date, datetime
from shutil import copy
import pickle
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import row, layout
from bokeh.models import ColumnDataSource, CustomJS, DataTable, TableColumn, RadioButtonGroup, DatePicker, Select
from bokeh.plotting import show


def write_results(area_res, out_folder, str_sum):
    out_date_folder = out_folder + "{}/".format(str(date.today()))
    out_date_time_folder = out_date_folder + "{}/"\
        .format(str(datetime.now().time()).replace(":", "-").replace(".", "-"))

    if not path.exists(out_folder):
        mkdir(out_folder)
    if not path.exists(out_date_folder):
        mkdir(out_date_folder)
    if not path.exists(out_date_time_folder):
        mkdir(out_date_time_folder)

    with open(out_date_time_folder + "summary" + '.txt', 'w+') as f:
        f.write(str_sum)
    f.close()

    with open(out_date_time_folder + "area_output" + '.pkl', 'wb+') as f:
        pickle.dump(area_res, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # copy the generated data
    copy('../data/area.pkl', out_date_time_folder + "area_input.pkl")
    copy('../data/households.pkl', out_date_time_folder + "households_input.pkl")

    return out_date_time_folder


def view_results(area_res, num_periods, out_date_time_folder):

    def dict_to_pd_dt(target_dict, k0_ks, k1_ks):
        for k0 in k0_ks:
            if k0 not in target_dict:
                target_dict[k0] = dict()
            for k1 in k1_ks:
                df = pd.DataFrame.from_dict(area_res[k0][k1], orient='index')
                df.to_csv(out_date_time_folder + "{}-{}.csv".format(k0, k1))
                target_dict[k0][k1] = df
                # target_dict[k0][k1].stack().reset_index()
        return target_dict

    def combine_dict_to_pd_dt(target_dict, k0_ks, k1_ks):
        for k1 in k1_ks:
            combined_dict = dict()
            for k0 in k0_ks:
                combined_dict[k0] = area_res[k0][k1]
            df = pd.DataFrame(combined_dict, columns=k0_ks)
            df.to_csv(out_date_time_folder + "{}-{}.csv".format("others", k1))
            target_dict[k1] = df

        return target_dict

    prices_dict = dict()
    k0_keys = [k0_profile]
    k1_keys = [k1_optimal, k1_heuristic]
    prices_dict = dict_to_pd_dt(prices_dict, k0_keys, k1_keys)

    demands_prices_fw_dict = dict()
    k0_keys = [k0_profile, k0_prices]
    k1_keys = [k1_optimal_fw, k1_heuristic_fw]
    demands_prices_fw_dict = dict_to_pd_dt(demands_prices_fw_dict, k0_keys, k1_keys)

    others_combined_dt = dict()
    k0_keys = [k0_demand_max, k0_par, k0_obj, k0_cost, k0_penalty, k0_ss]
    k1_keys = [k1_optimal_fw, k1_heuristic_fw]
    others_combined_dt = combine_dict_to_pd_dt(others_combined_dt, k0_keys, k1_keys)

    # a date picker for choosing the date of the results
    date_default = str(date.today())
    date_min = "2020-09-01"
    date_picker = DatePicker(title='Select date:', value=date_default, min_date=date_min, max_date=date_default)

    # a dropdown list for choosing the time of the results
    selected_date = date_picker.value
    select_options = [dirs for root, dirs, _ in walk("results/{}".format(selected_date)) if dirs != []][0]
    select = Select(title="Select time:", value=select_options[-1], options=select_options)

    # a data table for displaying numbers
    source = ColumnDataSource(others_combined_dt[k1_optimal_fw])
    columns = [TableColumn(field=str(i), title=str(i).replace("_", " ").capitalize()) for i in k0_keys]
    data_table = DataTable(source=source, columns=columns)

    # radio buttons for choosing the date source
    LABELS = ["Demand Profile", "Price Profile", "Others"]
    radio_button_group = RadioButtonGroup(labels=LABELS, active=0)

    def update_select_options(attr, old, new):
        select_opt = [dirs for root, dirs, _ in walk("results/{}".format(new)) if dirs != []][0]
        select.options = select_opt

    date_picker.on_change("value", update_select_options)
    select.js_on_change("value", CustomJS(code="""
        console.log('select: value=' + this.value, this.toString())
    """))
    radio_button_group.js_on_click(CustomJS(code="""
        console.log('radio_button_group: active=' + this.active, this.toString())
    """))

    # draw layout
    controls = row(date_picker, radio_button_group)
    l = layout([
        [date_picker, select],
        [radio_button_group],
        [data_table]

    ], sizing_mode='scale_width')

    # show graphs
    curdoc().add_root(l)
    show(l)


def test():
    date_folder = "2020-09-26"
    time_folder = "18-15-05-270068"
    date_time_folder = "results/{}/{}/".format(date_folder, time_folder)
    with open(date_time_folder + "area_output.pkl", 'rb') as f2:
        area_t = pickle.load(f2)
        f2.close()
    view_results(area_t, no_periods, date_time_folder)


# test()




