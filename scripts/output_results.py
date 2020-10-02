from os import path, mkdir
from datetime import date, datetime
from shutil import copy
import pickle
import pandas as pd
from scripts.input_parameter import *
from json import dumps


def aggregate_results(res, summary):
    # total time

    # total iterations

    # key input parameters - care factor weight, demand level scaler,

    # cost reduction in %

    # par reduction in %

    # max reduction in %

    # cost function type

    return summary


def write_results(households, area_res, out_folder, sum_dict, note, alg_labels):
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
        f.write(note)
    f.close()

    with open(out_date_time_folder + "area_output" + '.pkl', 'wb+') as f:
        pickle.dump(area_res, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    pd.DataFrame.from_dict(households, orient='index')\
        .to_csv(out_date_time_folder + "{}.csv".format("households"))

    sum_dict = aggregate_results(area_res, sum_dict)
    with open(out_date_time_folder + "summary" + '.pkl', 'wb+') as f:
        pickle.dump(sum_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    def dict_to_pd_dt(k0_ks, k1_ks):
        for k0 in k0_ks:
            if k0 in area_res:
                for k1 in k1_ks:
                    if k1 in area_res[k0]:
                        df = pd.DataFrame.from_dict(area_res[k0][k1], orient='index')
                        df.to_csv(out_date_time_folder + "{}-{}.csv".format(k0, k1))

    def combine_dict_to_pd_dt(k0_ks, k1_schedule, k1_pricing):
        for k1_s, k1_p in zip(k1_schedule, k1_pricing):
            combined_dict = dict()
            for k0 in k0_ks:
                if k0 in area_res:
                    if k1_p in area_res[k0]:
                        combined_dict[k0] = area_res[k0][k1_p]
                    elif k1_s in area_res[k0] and k0 == k0_time:
                        combined_dict[k0] = area_res[k0][k1_s]
            df = pd.DataFrame(combined_dict, columns=k0_ks)
            df.to_csv(out_date_time_folder + "{}-{}.csv".format("others", k1_p))

    k1_scheduling_keys = []
    k1_pricing_fw_keys = []
    for v in alg_labels.values():
        k1_scheduling_keys.append(v[k2_scheduling])
        k1_pricing_fw_keys.append(v[k2_pricing])

    k0_keys = [k0_demand]
    dict_to_pd_dt(k0_keys, k1_scheduling_keys)

    k0_keys = [k0_demand, k0_prices]
    dict_to_pd_dt(k0_keys, k1_pricing_fw_keys)

    k0_keys = [k0_demand_max, k0_demand_total, k0_par, k0_obj, k0_cost, k0_penalty, k0_step, k0_time]
    combine_dict_to_pd_dt(k0_keys, k1_scheduling_keys, k1_pricing_fw_keys)

    # copy the generated data
    copy('data/area.pkl', out_date_time_folder + "area_input.pkl")
    copy('data/households.pkl', out_date_time_folder + "households_input.pkl")

    return out_date_time_folder





