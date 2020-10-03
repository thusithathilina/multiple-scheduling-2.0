from scripts.iteration import *
from scripts.input_parameter import *


def experiment(num_households, num_tasks_min, new_data, cost_type, algorithms_labels):

    # -------------------- 0. initialise experiment (iteration = 0) -------------------- #
    print("---------- Experiment Summary ----------")
    str_note = "{0} households, min {1} tasks per household, {2} cost function, {3} care factor weight" \
        .format(num_households, num_tasks_min, cost_type, care_f_weight)
    print(str_note)

    print("---------- Experiments begin! ----------")
    # 0.1 - generation household data and the total preferred demand profile
    if new_data:
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods, file_household_area_folder,
                                           num_households, num_tasks_min, care_f_weight, care_f_max,
                                           max_demand_multiplier, file_probability, file_demand_list, algorithms_labels)
        print("Household data created...")
    else:
        households, area = area_read(file_household_area_folder)
        print("Household data read...")

    # 0.2 - read the model file, solver choice and the pricing table (price levels and the demand table)
    k1_temp = list(area[k0_demand_max].keys())[0]
    demand_level_scale = area[k0_demand_max][k1_temp][0] * pricing_table_weight
    models, solvers, pricing_table \
        = read_data(file_cp_pre, file_cp_ini, file_pricing_table, demand_level_scale, zero_digit)
    print("Pricing table created...")

    # 1 - the iteration!
    iterations = 0
    for alg in algorithms_labels.values():
        print("---------- {} ----------".format(alg))
        area, str_summary, iterations \
            = iteration(num_tasks_min, area, households, pricing_table, cost_type, str_note, solvers, models, alg)

    # -------------------- 4. process results -------------------- #
    key_parameters = {k0_tasks_no: num_tasks_min,
                      k0_households_no: num_households,
                      k0_penalty_weight: care_f_weight,
                      k0_cost_type: cost_function_type}

    output_date_time_folder = write_results(iterations, key_parameters, area, output_folder,
                                            str_note, algorithms_labels)
