from shutil import copy
import pickle
import pandas as pd
from scripts.input_parameter import *
from scripts.cfunctions import average
from pathlib import Path


def aggregate_results(area_dt, key_params):

    def reduction(k0):
        for alg2 in area_dt:
            init_value = area_dt[alg2][k0][0]
            final_value = list(area_dt[alg2][k0].values())[-1]
            alg_reduction = (init_value - final_value) / init_value
            summary[alg2][k0 + " initial"] = init_value
            summary[alg2][k0 + " final"] = final_value
            summary[alg2][k0 + " reduction"] = "{:.2%}".format(round(alg_reduction, 3))

    summary = dict()
    for alg in area_dt:
        all_run_times = list(area_dt[alg][k0_time].values())[1:]
        summary[alg] = dict()
        summary[alg]["Average " + k0_time] = round(average(all_run_times), 3)
        summary[alg]["No_iterations"] = len(all_run_times)
        for param_k, param_v in key_params.items():
            summary[alg][param_k] = param_v

    reduction(k0_demand_max)
    reduction(k0_par)
    reduction(k0_obj)
    reduction(k0_cost)
    reduction(k0_par)

    return area_dt, summary


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

    def save_to_csv(data_dict, file_name):
        pd.DataFrame.from_dict(data_dict, orient='index').to_csv(exp_folder + "{}.csv".format(file_name))

    save_to_csv(sum_dict, "summary")
    for alg in area_res:
        save_to_csv(area_res[alg][k0_demand], "{}-{}".format(alg, k0_demand))
        save_to_csv(area_res[alg][k0_prices], "{}-{}".format(alg, k0_prices))
        others_keys = {k0_demand_max, k0_demand_total, k0_par, k0_obj, k0_cost, k0_penalty, k0_step, k0_time}
        others_dict = {k: area_res[alg][k] for k in area_res[alg].keys() & others_keys}
        save_to_csv(others_dict, "{}-{}".format(alg, "others"))

    # copy the generated data
    copy('data/area.pkl', exp_folder + "area_input.pkl")
    copy('data/households.pkl', exp_folder + "households_input.pkl")

    return sum_dict



