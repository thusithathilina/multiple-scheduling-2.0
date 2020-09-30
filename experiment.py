from scripts.iteration import *
from scripts.input_parameter import *


def experiment(num_households, num_tasks, new_data, cost_type):

    # -------------------- 0. initialise experiment (iteration = 0) -------------------- #
    print("---------- Experiment Summary ----------")
    str_summary = "{0} households, {1} tasks per household, {2} cost function" \
        .format(num_households, num_tasks, cost_type)
    print(str_summary)
    print("---------- Experiments begin! ----------")

    # 0.0 algorithm choices
    algorithms_labels = dict()
    algorithms_labels[k1_optimal] = dict()
    algorithms_labels[k1_optimal][k2_scheduling] = k1_optimal
    algorithms_labels[k1_optimal][k2_pricing] = "{}_fw".format(k1_optimal)

    algorithms_labels[k1_heuristic] = dict()
    algorithms_labels[k1_heuristic][k2_scheduling] = k1_heuristic
    algorithms_labels[k1_heuristic][k2_pricing] = "{}_fw".format(k1_heuristic)

    # 0.1 - generation household data and the total preferred demand profile
    if new_data:
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods, file_household_area_folder,
                                           num_households, num_tasks, care_f_weight, care_f_max, max_demand_multiplier,
                                           file_probability, file_demand_list, algorithms_labels)
        print("Household data created...")
    else:
        households, area = area_read(file_household_area_folder)
        print("Household data read...")
    area[k0_cost_type] = cost_type

    # 0.2 - read the model file, solver choice and the pricing table (price levels and the demand table)
    k1_temp = list(area[k0_demand_max].keys())[0]
    demand_level_scale = area[k0_demand_max][k1_temp][0] * pricing_table_weight
    models, solvers, pricing_table \
        = read_data(file_cp_pre, file_cp_ini, file_pricing_table, demand_level_scale, zero_digit)
    print("Pricing table created...")

    for alg in algorithms_labels.values():
        print("---------- {} ----------".format(alg))
        area, str_summary \
            = iteration(num_tasks, area, households, pricing_table, cost_type, str_summary, solvers, models, alg)

    # -------------------- 4. process results -------------------- #
    output_date_time_folder = write_results(area, output_folder, str_summary, algorithms_labels)
