from os import path, mkdir
from datetime import date, datetime
from shutil import copy
import pickle
from bokeh.models import ColumnDataSource


def write_results(area, out_folder):
    out_date_folder = out_folder + "{}/".format(str(date.today()))
    out_date_time_folder = out_date_folder + "{}/".format(str(datetime.now().time()).replace(":", "-"))

    if not path.exists(out_folder):
        mkdir(out_folder)
    if not path.exists(out_date_folder):
        mkdir(out_date_folder)
    if not path.exists(out_date_time_folder):
        mkdir(out_date_time_folder)

    with open(out_date_time_folder + "area_ouput" + '.pkl', 'wb+') as f:
        pickle.dump(area, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    # copy the generated data
    copy('data/area.pkl', out_date_time_folder + "area_input.pkl")
    copy('data/households.pkl', out_date_time_folder + "households_input.pkl")

    return out_date_time_folder


def view_results(area_res, out_date_time_folder):

    y_values = 0

    # demand profile per iteration
    data_demand_profiles = {}

    # prices per iteration

    # objective value per iteration

    # cost per iteration

    # penalty per iteration

    return True


date_folder = "2020-09-25"
time_folder = "19-07-07.766730"
date_time_folder = "results/{}/{}/".format(date_folder, time_folder)
with open("data/area.pkl", 'rb') as f:
    area = pickle.load(f)
    f.close()
view_results(area, date_time_folder)




