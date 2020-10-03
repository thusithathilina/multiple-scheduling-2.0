from os import path, mkdir
from datetime import date, datetime
from shutil import copy
import pickle
import pandas as pd
from scripts.input_parameter import *
from scripts.cfunctions import average
from json import dumps


def aggregate_results(itr, res, key_params):

    def reduction(k0):
        summary[k0 + " reduction"] = dict()
        for alg2 in res[k0]:
            summary[k0 + " reduction"][alg2] \
                = round((res[k0][alg2][0] - res[k0][alg2][itr]) / res[k0][alg2][0], 2)

    def save_key_parameters():
        for param_k, param_v in key_params.items():
            summary[param_k] = dict()
            for alg3 in res[k0_demand]:
                summary[param_k][alg3] = param_v

    summary = dict()
    save_key_parameters()
    reduction(k0_demand_max)
    reduction(k0_par)
    reduction(k0_obj)
    reduction(k0_cost)
    reduction(k0_par)

    summary["Average " + k0_time] = dict()
    for alg in res[k0_time]:
        summary["Average " + k0_time][alg] = average(res[k0_time][alg].values())

    return res, summary


def write_results(iterations, key_parameters, area_res, out_folder, note, alg_labels):

    area_res, sum_dict = aggregate_results(iterations, area_res, key_parameters)

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

    with open(out_date_time_folder + "summary" + '.pkl', 'wb+') as f:
        pickle.dump(sum_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    pd.DataFrame.from_dict(sum_dict, orient='index').to_csv(out_date_time_folder + "{}.csv".format("summary"))

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
            pd_columns = []
            for k0 in k0_ks:
                if k0 in area_res:
                    if k1_s in area_res[k0] and k0 == k0_time:
                        combined_dict[k1_time_scheduling] = area_res[k0][k1_s]
                        combined_dict[k1_time_pricing] = area_res[k0][k1_p]
                        pd_columns.extend([k1_time_scheduling, k1_time_pricing])
                    elif k1_p in area_res[k0]:
                        combined_dict[k0] = area_res[k0][k1_p]
                        pd_columns.append(k0)
            df = pd.DataFrame(combined_dict, columns=pd_columns)
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
    # pd.DataFrame.from_dict(households, orient='index') \
    #     .to_csv(out_date_time_folder + "{}.csv".format("households"))

    return out_date_time_folder





