from more_itertools import grouper
from multiple.data_generation import *
from multiple.household_scheduling import *
from multiple.drsp_pricing import *
from multiple.input_parameter import *
from multiple.output_results import write_results
from time import strftime, localtime
import timeit


def iteration():
    def update_area_trackers(i, result_type, demand_profile, obj, cost, penalty, step_size, prices):
        area[k0_obj][result_type][i] = obj
        area[k0_cost][result_type][i] = cost
        area[k0_penalty][result_type][i] = penalty
        area[k0_ss][result_type][i] = step_size
        area[k0_prices][result_type][i] = prices
        area[k0_profile][result_type][i] = demand_profile

    # 0 - initialise experiment (iteration = 0)
    itr = 0
    print("---------- Experiment Summary ----------")
    print("{0} households, {1} tasks per household".format(no_households, no_tasks))
    print("---------- Experiments begin! ----------")

    # 0.1 - generation household data and the total preferred demand profile
    if new_households:
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods, file_household_area_folder,
                                           no_households, no_tasks, care_f_weight, care_f_max, max_demand_multiplier)
        print("Household data created...")
    else:
        households, area = area_read(file_household_area_folder)
        print("Household data read...")
    area_demand_profile = area[k0_profile][k1_optimal][0]
    area_demand_max = max(area_demand_profile)

    # 0.2 - read the model file, solver choice and the pricing table (price levels and the demand table)
    demand_level_scale = area_demand_max * pricing_table_weight
    models, solvers, pricing_table \
        = read_data(file_cp_pre, file_cp_ini, file_pricing_table, demand_level_scale, zero_digit)
    solver_choice = solvers[solver_type]
    model_file = models[solver_type][model_type]
    print("Model, solver and pricing data created...")

    # 0.3 - the prices of the total preferred demand profile
    heuristic_demand_profile_updated, heuristic_step_size, heuristic_prices, heuristic_cost, \
        optimal_demand_profile_updated, optimal_step_size, optimal_prices, optimal_cost \
        = pricing_master_problem(itr, pricing_table, area, cost_type)
    print("First day prices calculated...")

    # 0.4 - initialise tracker values
    update_area_trackers(itr, k1_optimal_fw, optimal_demand_profile_updated,
                         optimal_cost, optimal_cost, 0, optimal_step_size, optimal_prices)
    update_area_trackers(itr, k1_heuristic_fw, heuristic_demand_profile_updated,
                         heuristic_cost, heuristic_cost, 0, heuristic_step_size, heuristic_prices)

    print("---------- Initialisation done! ----------")

    # 1 ~ 2 - rescheduling and pricing (iteration = k where k > 0)
    itr = 1
    total_runtime_heuristic = 0
    total_runtime_optimal = 0
    while optimal_step_size > 0.01:

        # 1.1 - reschedule given the prices at iteration k - 1
        heuristic_area_profile_scheduling = [0] * no_intervals
        optimal_area_profile_scheduling = [0] * no_intervals
        for key, household in households.items():
            heuristic_starts, heuristic_profile_scheduling, heuristic_obj, heuristic_runtime, \
            optimal_starts, optimal_profile_scheduling, optimal_obj, optimal_runtime \
                = household_scheduling_subproblem(no_intervals, no_tasks, no_periods, no_intervals_periods,
                                                  household, care_f_weight, care_f_max, area[k0_prices], itr,
                                                  model_file, model_type, solver_type,
                                                  solver_choice, var_selection, val_choice)

            heuristic_area_profile_scheduling \
                = [x + y for x, y in zip(heuristic_profile_scheduling, heuristic_area_profile_scheduling)]
            optimal_area_profile_scheduling \
                = [x + y for x, y in zip(optimal_profile_scheduling, optimal_area_profile_scheduling)]
            total_runtime_heuristic += heuristic_runtime
            total_runtime_optimal += optimal_runtime
            print("household {}".format(key))

        # 1.2 - aggregate rescheduled demand profile for the pricing purpose
        area[k0_profile][k1_heuristic][itr] \
            = [sum(x) for x in grouper(heuristic_area_profile_scheduling, no_intervals_periods)]
        area[k0_profile][k1_optimal][itr] \
            = [sum(x) for x in grouper(optimal_area_profile_scheduling, no_intervals_periods)]

        # 2.1 - pricing
        heuristic_demand_profile_updated, heuristic_step_size, heuristic_prices, heuristic_cost, \
        optimal_demand_profile_updated, optimal_step_size, optimal_prices, optimal_cost \
            = pricing_master_problem(itr, pricing_table, area, cost_type)
        print("step size at iteration {}: optimal = {}, heuristic = {}"
              .format(itr, float(optimal_step_size), float(heuristic_step_size)))

        # 2.2 - update the demand profiles, prices and the step size
        update_area_trackers(itr, k1_optimal_fw, optimal_demand_profile_updated,
                             optimal_cost, optimal_cost, 0, optimal_step_size, optimal_prices)
        update_area_trackers(itr, k1_heuristic_fw, heuristic_demand_profile_updated,
                             heuristic_cost, heuristic_cost, 0, heuristic_step_size, heuristic_prices)

        # 3 - move on to the next iteration
        itr += 1

    print("---------- Result Summary ----------")
    print("Converged in {0} iteration".format(itr))
    print("---------- Iteration done! ----------")

    # 4 - process results
    output_date_time_folder = write_results(area, output_folder)
    # view_results(area, output_date_time_folder)


iteration()
