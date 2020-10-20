from scripts.csv_to_dict_converter import convert_from_csv
from scripts.data_mode import DataMode
from scripts.fdia_integrator import fdia_inject, fdia_analyse_impact
from scripts.iteration import *
from scripts.input_parameter import *
from scripts.output_results import write_results


def experiment(num_households, num_tasks_min, num_tasks_max, data_mode, cost_type, algorithms_labels, experiment_folder,
               job_file=None, enable_fdia=False, injection_percentage=0, attack_result_file_prepend=''):

    print("---------- Experiment Summary ----------")
    str_note = "{0} households, min {1} tasks per household, {2} cost function, {3} care factor weight" \
        .format(num_households, num_tasks_min, cost_type, care_f_weight)
    print(str_note)

    print("---------- Data Generation ----------")
    if data_mode == DataMode.CREATE:
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods,
                                           file_household_area_folder, experiment_folder,
                                           num_households, num_tasks_min, num_tasks_max, care_f_weight, care_f_max,
                                           max_demand_multiplier, file_probability, file_demand_list, algorithms_labels)
        print("Household data created...")
    elif data_mode == DataMode.EXISTING:
        households, area = area_read(file_household_area_folder)
        print("Household data read...")
    else:
        households, area = convert_from_csv(job_file, algorithms_labels, no_intervals_periods)

    k1_temp = list(algorithms_labels)[0]
    demand_level_scale = area[k1_temp][k0_demand_max][0] * pricing_table_weight
    models, solvers, pricing_table \
        = read_data(file_cp_pre, file_cp_ini, file_pricing_table, demand_level_scale, zero_digit)
    print("Pricing table created...")

    # FDIA injection
    if enable_fdia and injection_percentage > 0:
        households, area = fdia_inject(households, area, list(algorithms_labels)[0], injection_percentage)

    for alg in algorithms_labels.values():
        print("---------- Iteration Begin! ----------")
        area, str_summary, num_iterations \
            = iteration(area, households, pricing_table, cost_type, str_note, solvers, models, alg)
        print("---------- Iteration Done! ----------")
        print("Converged in {0} iteration".format(num_iterations))

    print("---------- Results ----------")
    key_parameters = {k0_households_no: num_households,
                      k0_tasks_no: num_tasks_min,
                      k0_penalty_weight: care_f_weight,
                      k0_cost_type: cost_function_type}
    print("Key parameters saved...")

    exp_summary = write_results(key_parameters, area, experiment_folder, str_note)
    print("Results saved to files...")

    if enable_fdia:
        fdia_analyse_impact(households, area, list(algorithms_labels)[0], num_iterations, pricing_table,
                            cost_function_type, injection_percentage, attack_result_file_prepend)

    return exp_summary

