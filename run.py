from more_itertools import grouper
from multiple.data_generation import *
from multiple.household_scheduling import *
from multiple.drsp_pricing import *
from multiple.fixed_parameter import *
from time import strftime, localtime
import timeit

# time related parameters
no_intervals = 48
no_periods = 48
no_intervals_periods = int(no_intervals / no_periods)

# household related parameters
new_households = True
new_households = False
no_households = 100
no_tasks = 5
max_demand_multiplier = no_tasks
care_f_max = 10
care_f_weight = 100

# pricing related parameters
pricing_table_weight = 1.0
cost_type = "linear"  # or "piece-wise linear"

# solver related parameters
var_selection = "smallest"
val_choice = "indomain_min"
model_type = "pre"
solver_type = "cp"

# external file related parameters
file_cp_pre = 'models/Household-cp-pre.mzn'
file_cp_ini = 'models/Household-cp.mzn'
file_pricing_table = 'inputs/pricing_table_0.csv'
file_household_area_folder = 'data/'


def iteration():
    # 0 - initialise experiment (iteration = 0)
    itr = 0
    print("---------- Summary ----------")
    print("{0} households, {1} tasks per household".format(no_households, no_tasks))
    print("---------- Experiments begin! ----------")

    # 0 - generation household data and the total preferred demand profile
    if new_households:
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods, file_household_area_folder,
                                           no_households, no_tasks, care_f_weight, care_f_max, max_demand_multiplier)
        print("Household data created...")
    else:
        households, area = area_read(file_household_area_folder)
        print("Household data read...")
    area_demand_profile = area[k0_profile][k1_period][0]
    area_demand_max = max(area_demand_profile)

    # 0 - read the model file, solver choice and the pricing table (price levels and the demand table)
    models, solvers, pricing_table \
        = read_data(file_cp_pre, file_cp_ini, file_pricing_table, area_demand_max, pricing_table_weight)
    solver_choice = solvers[solver_type]
    model_file = models[solver_type][model_type]
    print("Model, solver and pricing data created...")

    # 0 - the prices of the total preferred demand profile
    heuristic_demand_profile_updated, heuristic_best_step_size, heuristic_price_day, heuristic_cost, \
        optimal_demand_profile_updated, optimal_best_step_size, optimal_price_day, optimal_cost \
        = pricing_master_problem(itr, pricing_table, area, cost_type)
    print("First day prices calculated...")

    # 0 - initialise tracker values
    area[k0_obj][k1_optimal][0] = optimal_cost
    area[k0_obj][k1_heuristic][0] = optimal_cost
    area[k0_cost][k1_optimal][0] = optimal_cost
    area[k0_cost][k1_heuristic][0] = optimal_cost
    area[k0_inconvenient][k1_optimal][0] = 0
    area[k0_inconvenient][k1_heuristic][0] = 0
    area[k0_ss][k1_optimal][0] = optimal_best_step_size
    area[k0_ss][k1_heuristic][0] = heuristic_best_step_size
    area[k0_price_history][k1_optimal][0] = optimal_price_day
    area[k0_price_history][k1_heuristic][0] = heuristic_price_day

    print("---------- Initialisation done! ----------")

    # 1 - rescheduling and pricing (iteration = k where k > 0)
    itr = 1
    total_runtime_heuristic = 0
    total_runtime_optimal = 0
    while True:

        # 1 - reschedule given the prices at iteration k - 1
        prices_pre = area[k0_price_history][k1_optimal][itr - 1]
        area_demand_profile_heuristic = [0] * no_intervals
        area_demand_profile_optimal = [0] * no_intervals
        for key, household in households.items():
            heuristic_starts, heuristic_profile, heuristic_obj, heuristic_runtime, \
                optimal_starts, optimal_profile, optimal_obj, optimal_runtime \
                = household_scheduling_subproblem(no_intervals, no_tasks, no_periods, no_intervals_periods,
                                                  household, care_f_weight, care_f_max, prices_pre,
                                                  model_file, model_type, solver_type,
                                                  solver_choice, var_selection, val_choice)

            area_demand_profile_heuristic = [x + y for x, y in zip(heuristic_profile, area_demand_profile_heuristic)]
            area_demand_profile_optimal = [x + y for x, y in zip(optimal_profile, area_demand_profile_optimal)]
            total_runtime_heuristic += heuristic_runtime
            total_runtime_optimal += optimal_runtime
            print("household {}".format(key))

        # 1 - aggregate demand profile
        area[k0_profile][k1_heuristic][itr] = [sum(x) for x in grouper(area_demand_profile_heuristic,
                                                                       no_intervals_periods)]
        area[k0_profile][k1_optimal][itr] = [sum(x) for x in grouper(area_demand_profile_optimal,
                                                                     no_intervals_periods)]

        # 1 - pricing
        heuristic_demand_profile_updated, heuristic_best_step_size, heuristic_price_day, heuristic_cost, \
            optimal_demand_profile_updated, optimal_best_step_size, optimal_price_day, optimal_cost \
            = pricing_master_problem(itr, price_levels, demand_table, area, cost_type)

        # 1 - update the prices and the step size


        # 1 - move on to the next iteration
        itr += 1


iteration()
