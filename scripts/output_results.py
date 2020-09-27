from os import path, mkdir
from datetime import date, datetime
from shutil import copy
import pickle
import pandas as pd
from multiple.scripts.input_parameter import *


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

    def dict_to_pd_dt(k0_ks, k1_ks):
        for k0 in k0_ks:
            if k0 in area_res:
                for k1 in k1_ks:
                    if k1 in area_res[k0]:
                        df = pd.DataFrame.from_dict(area_res[k0][k1], orient='index')
                        df.to_csv(out_date_time_folder + "{}-{}.csv".format(k0, k1))

    def combine_dict_to_pd_dt(k0_ks, k1_ks):
        for k1 in k1_ks:
            combined_dict = dict()
            for k0 in k0_ks:
                if k0 in area_res:
                    combined_dict[k0] = area_res[k0][k1]
            df = pd.DataFrame(combined_dict, columns=k0_ks)
            df.to_csv(out_date_time_folder + "{}-{}.csv".format("others", k1))

    k0_keys = [k0_demand]
    k1_keys = [k1_optimal_scheduling, k1_heuristic_scheduling]
    dict_to_pd_dt(k0_keys, k1_keys)

    k0_keys = [k0_demand, k0_prices]
    k1_keys = [k1_optimal_fw, k1_heuristic_fw]
    dict_to_pd_dt(k0_keys, k1_keys)

    k0_keys = [k0_demand_max, k0_demand_total, k0_par, k0_obj, k0_cost, k0_penalty, k0_step, k0_time]
    k1_keys = [k1_optimal_fw, k1_heuristic_fw]
    combine_dict_to_pd_dt(k0_keys, k1_keys)

    # copy the generated data
    copy('data/area.pkl', out_date_time_folder + "area_input.pkl")
    copy('data/households.pkl', out_date_time_folder + "households_input.pkl")

    return out_date_time_folder





