from scripts.data_generation import *
from scripts.household_scheduling import *
from scripts.drsp_pricing import *
from scripts.input_parameter import *
from scripts.output_results import write_results
from scripts.cfunctions import *


def update_data(r_dict, itr, k1_alg, d, k0):
    if d is not None:
        r_dict[k0][k1_alg][itr] = d
    return r_dict


def update_area_data(area_dict, i, k1_algorithm, demands, prices, obj, cost, penalty, step_size, time):
    if demands is not None:
        max_demand = max(demands)
        area_dict[k0_demand][k1_algorithm][i] = demands
        area_dict[k0_demand_max][k1_algorithm][i] = max_demand
        area_dict[k0_demand_total][k1_algorithm][i] = sum(demands)
        area_dict[k0_par][k1_algorithm][i] = round(average(demands) / max_demand, 2)

    area_dict = update_data(area_dict, i, k1_algorithm, prices, k0_prices)
    area_dict = update_data(area_dict, i, k1_algorithm, obj, k0_obj)
    area_dict = update_data(area_dict, i, k1_algorithm, cost, k0_cost)
    area_dict = update_data(area_dict, i, k1_algorithm, penalty, k0_penalty)
    area_dict = update_data(area_dict, i, k1_algorithm, step_size, k0_step)
    area_dict = update_data(area_dict, i, k1_algorithm, time, k0_time)

    return area_dict


def iteration(num_households, num_tasks, new_data):

    def extract_pricing_results(k1_algorithm_scheduling, k1_algorithm_fw, results):
        prices = results[k1_algorithm_scheduling][k0_prices]
        cost = results[k1_algorithm_scheduling][k0_cost]

        demands_fw = results[k1_algorithm_fw][k0_demand]
        prices_fw = results[k1_algorithm_fw][k0_prices]
        cost_fw = results[k1_algorithm_fw][k0_cost]
        penalty_fw = results[k1_algorithm_fw][k0_penalty]
        step_fw = results[k1_algorithm_fw][k0_step]
        return prices, cost, demands_fw, prices_fw, cost_fw, penalty_fw, step_fw

    def extract_rescheduling_results(k1_algorithm_scheduling, results):
        demands_new = results[k1_algorithm_scheduling][k0_demand]
        obj = results[k1_algorithm_scheduling][k0_obj]
        penalty = results[k1_algorithm_scheduling][k0_penalty]
        time = results[k1_algorithm_scheduling][k0_time]
        return demands_new, obj, penalty, time

    # -------------------- 0. initialise experiment (iteration = 0) -------------------- #
    print("---------- Experiment Summary ----------")
    str_summary = "{0} households, {1} tasks per household, {2} cost function"\
        .format(num_households, num_tasks, cost_type)
    print(str_summary)
    print("---------- Experiments begin! ----------")

    # 0.1 - generation household data and the total preferred demand profile
    if new_data:
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods, file_household_area_folder,
                                           num_households, num_tasks, care_f_weight, care_f_max, max_demand_multiplier,
                                           file_probability, file_demand_list)
        print("Household data created...")
    else:
        households, area = area_read(file_household_area_folder)
        print("Household data read...")

    # 0.2 - read the model file, solver choice and the pricing table (price levels and the demand table)
    demand_level_scale = area[k0_demand_max][k1_optimal_scheduling][0] * pricing_table_weight
    models, solvers, pricing_table \
        = read_data(file_cp_pre, file_cp_ini, file_pricing_table, demand_level_scale, zero_digit)
    print("Pricing table created...")

    # 0.3 - the prices of the total preferred demand profile
    pricing_results = pricing_master_problem(0, pricing_table, area, cost_type)
    print("First day prices calculated...")

    # 0.4 - initialise tracker values
    # 0.4.1 - heuristic: initialise the demand profile, prices,
    # objective values, costs, penalties, steps and the run times
    heuristic_prices, heuristic_cost, heuristic_demands_fw, heuristic_prices_fw, \
    heuristic_cost_fw, heuristic_penalty_fw, heuristic_step_fw \
        = extract_pricing_results(k1_heuristic_scheduling, k1_heuristic_fw, pricing_results)
    area = update_area_data(area, 0, k1_heuristic_scheduling, None, heuristic_prices,
                            heuristic_cost, heuristic_cost, None, None, 0)
    area = update_area_data(area, 0, k1_heuristic_fw, heuristic_demands_fw, heuristic_prices_fw,
                            heuristic_cost_fw, heuristic_cost_fw, heuristic_penalty_fw, heuristic_step_fw, 0)

    # 0.4.2 - optimal: initialise the demand profile, prices,
    # objective values, costs, penalties, steps and the run times
    optimal_prices, optimal_cost, optimal_demands_fw, optimal_prices_fw, \
    optimal_cost_fw, optimal_penalty_fw, optimal_step_fw \
        = extract_pricing_results(k1_optimal_scheduling, k1_optimal_fw, pricing_results)
    area = update_area_data(area, 0, k1_optimal_scheduling, None, optimal_prices,
                            optimal_cost, optimal_cost, 0, None, 0)
    area = update_area_data(area, 0, k1_optimal_fw, optimal_demands_fw, optimal_prices_fw,
                            optimal_cost_fw + optimal_penalty_fw, optimal_cost_fw, optimal_penalty_fw, optimal_step_fw, 0)
    print("The demand profile, prices, the cost, the step, and the run time initialised...")

    print("---------- Initialisation done! ----------")

    # iteration = k where k > 0
    itr = 1
    solver_choice = solvers[solver_type]
    model_file = models[solver_type][model_type]

    total_time_scheduling_heuristic = 0
    total_time_pricing_heuristic = 0
    heuristic_step_fw = 1

    total_time_scheduling_optimal = 0
    total_time_pricing_optimal = 0
    optimal_step_fw = 1

    while optimal_step_fw > 0 or heuristic_step_fw > 0:

        # -------------------- 1. rescheduling step -------------------- #
        heuristic_demands_scheduling = [0] * no_intervals
        heuristic_obj_area = 0
        heuristic_penalty_area = 0

        optimal_demands_scheduling = [0] * no_intervals
        optimal_obj_area = 0
        optimal_penalty_area = 0

        # 1.1 - reschedule given the prices at iteration k - 1
        for key, household in households.items():

            # 1.1.1 - reschedule a household
            rescheduling_results \
                = household_scheduling_subproblem(no_intervals, num_tasks, no_periods, no_intervals_periods,
                                                  household, care_f_weight, care_f_max, area[k0_prices], itr,
                                                  model_file, model_type,
                                                  solver_type, solver_choice, var_selection, val_choice)

            # 1.1.2 - heuristic: extract results and update the demand profiles, total objective value and the runtime
            heuristic_demands_household, heuristic_obj_household, heuristic_penalty_household, heuristic_time_household \
                = extract_rescheduling_results(k1_heuristic_scheduling, rescheduling_results)
            heuristic_demands_scheduling \
                = [x + y for x, y in zip(heuristic_demands_household, heuristic_demands_scheduling)]
            heuristic_obj_area += heuristic_obj_household
            heuristic_penalty_area += heuristic_penalty_household
            total_time_scheduling_heuristic += heuristic_time_household

            # 1.1.2 - optimal: extract results and update the demand profiles, total objective value and the runtime
            optimal_demands_household, optimal_obj_household, optimal_penalty_household, optimal_time_household \
                = extract_rescheduling_results(k1_optimal_scheduling, rescheduling_results)
            optimal_demands_scheduling \
                = [x + y for x, y in zip(optimal_demands_household, optimal_demands_scheduling)]
            optimal_obj_area += optimal_obj_household
            optimal_penalty_area += optimal_penalty_household
            total_time_scheduling_optimal += optimal_time_household

            print("household {}".format(key))

        # 1.2 - process and save results
        # 1.2.1 - heuristic: convert the area demand profile for the pricing purpose
        # and save the converted area demand profiles, total objective value and the total rescheduling time
        heuristic_demands = [sum(x) for x in grouper(heuristic_demands_scheduling, no_intervals_periods)]
        area = update_area_data(area, itr, k1_heuristic_scheduling, heuristic_demands, None,
                                heuristic_obj_area, heuristic_obj_area - optimal_penalty_area, optimal_penalty_area,
                                None, total_time_scheduling_heuristic)

        # 1.2.2 - optimal: heuristic: convert the area demand profile for the pricing purpose
        # and save the converted area demand profiles, total objective value and the total rescheduling time
        optimal_demands = [sum(x) for x in grouper(optimal_demands_scheduling, no_intervals_periods)]
        area = update_area_data(area, itr, k1_optimal_scheduling, optimal_demands, None,
                                optimal_obj_area, optimal_obj_area - optimal_penalty_area, optimal_penalty_area,
                                None, total_time_scheduling_optimal)

        # -------------------- 2. pricing step -------------------- #
        # 2.1 - calculate the prices, the step size and the cost after applying the FW algorithm
        pricing_results_fw = pricing_master_problem(itr, pricing_table, area, cost_type)

        # 2.2 - save results
        # 2.2.1 - heuristic: save the demand profiles, prices and the step size at this iteration
        heuristic_prices, heuristic_cost, heuristic_demands_fw, heuristic_prices_fw, heuristic_cost_fw, \
        heuristic_penalty_fw, heuristic_step_fw \
            = extract_pricing_results(k1_heuristic_scheduling, k1_heuristic_fw, pricing_results_fw)
        # todo - handle if step == 0
        area = update_area_data(area, itr, k1_heuristic_scheduling,
                                None, heuristic_prices,
                                None, None, None, None, 0)
        area = update_area_data(area, itr, k1_heuristic_fw,
                                heuristic_demands_fw, heuristic_prices_fw,
                                heuristic_cost_fw + heuristic_penalty_fw, heuristic_cost_fw, heuristic_penalty_fw,
                                heuristic_step_fw, None)

        # 2.2.2 - optimal: save the demand profiles, prices and the step size at this iteration
        optimal_prices, optimal_cost, optimal_demand_fw, optimal_prices_fw, optimal_cost_fw, \
        optimal_penalty_fw, optimal_step_fw \
            = extract_pricing_results(k1_optimal_scheduling, k1_optimal_fw, pricing_results_fw)
        # todo - handle if step == 0
        area = update_area_data(area, itr, k1_optimal_scheduling,
                                None, optimal_prices,
                                None, None, None, None, None)
        area = update_area_data(area, itr, k1_optimal_fw,
                                optimal_demand_fw, optimal_prices_fw,
                                optimal_cost_fw + optimal_penalty_fw, optimal_cost_fw, optimal_penalty_fw,
                                optimal_step_fw, None)

        print("step size at iteration {}: heuristic = {}, optimal = {}"
              .format(itr, float(heuristic_step_fw), float(optimal_step_fw)))

        # 3 - move on to the next iteration
        itr += 1

    print("---------- Result Summary ----------")
    print("Converged in {0} iteration".format(itr - 1))
    print("---------- Iteration done! ----------")

    # -------------------- 4. process results -------------------- #
    output_date_time_folder = write_results(area, output_folder, str_summary)

