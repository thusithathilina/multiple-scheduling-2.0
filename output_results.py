from multiple.input_parameter import *
from os import path, mkdir
from datetime import date, datetime
from shutil import copy
import pickle
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, DataTable, Button, \
    TableColumn, Panel, Tabs, RadioButtonGroup
from bokeh.plotting import figure, show


def write_results(area_res, out_folder):
    out_date_folder = out_folder + "{}/".format(str(date.today()))
    out_date_time_folder = out_date_folder + "{}/"\
        .format(str(datetime.now().time()).replace(":", "-").replace(".", "-"))

    if not path.exists(out_folder):
        mkdir(out_folder)
    if not path.exists(out_date_folder):
        mkdir(out_date_folder)
    if not path.exists(out_date_time_folder):
        mkdir(out_date_time_folder)

    with open(out_date_time_folder + "area_output" + '.pkl', 'wb+') as f:
        pickle.dump(area_res, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # copy the generated data
    copy('data/area.pkl', out_date_time_folder + "area_input.pkl")
    copy('data/households.pkl', out_date_time_folder + "households_input.pkl")

    return out_date_time_folder


def view_results(area_res, num_periods, out_date_time_folder):

    columns = [TableColumn(field=str(i), title=str(i + 1)) for i in range(num_periods)]

    def dict_to_pd_dt(k0_ks, k1_ks):
        for k0 in k0_ks:
            if k0 not in df_dict:
                df_dict[k0] = dict()
                dt_dict[k0] = dict()
            for k1 in k1_ks:
                df_dict[k0][k1] = pd.DataFrame.from_dict(area_res[k0][k1], orient='index').stack().reset_index()
                df_dict[k0][k1].columns = ["iteration", "period", k0]
                # dt_dict[k0][k1] = DataTable(source=df_dict[k0][k1], columns=columns)

    df_dict = dict()
    dt_dict = dict()

    k0_keys = [k0_profile, k0_obj, k0_cost, k0_penalty]
    k1_keys = [k1_optimal, k1_heuristic, k1_optimal_fw, k1_heuristic_fw]
    dict_to_pd_dt(k0_keys, k1_keys)

    k0_keys = [k0_ss, k0_prices]
    k1_keys = [k1_optimal_fw, k1_heuristic_fw]
    dict_to_pd_dt(k0_keys, k1_keys)

    # show data tables
    LABELS = ["Demand Profile", "Prices", "Others"]
    radio_button_group = RadioButtonGroup(labels=LABELS, active=0)

    # show graphs
    show(radio_button_group)

    return True


date_folder = "2020-09-25"
time_folder = "23-39-21-240982"
date_time_folder = "results/{}/{}/".format(date_folder, time_folder)
with open(date_time_folder + "area_output.pkl", 'rb') as f2:
    area_t = pickle.load(f2)
    f2.close()
view_results(area_t, no_periods, date_time_folder)




