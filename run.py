from more_itertools import grouper
from bisect import bisect_left
from multiple.cfunctions import find_ge, find_le
from multiple.household_scheduling import *
from multiple.data_generation import *
from time import strftime, localtime
import timeit

# time related parameters
no_intervals = 144
no_periods = 48
no_intervals_periods = int(no_intervals / no_periods)

# household related parameters
new_households = True
# new_households = False
no_households = 1000
no_tasks = 5
max_demand_multiplier = no_tasks
care_f_max = 10
care_f_weight = 100

# demand profile related parameters
dp_profile = "profile"
dp_interval = "interval"
dp_period = "period"
dp_optimal = "optimal"
dp_heuristic = "heuristic"

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


def pricing_cost(demand_profile, price_levels, demand_table, cost_function_type):
    price_day = []
    cost = 0
    for d, demand_level in zip(demand_profile, demand_table):
        level = bisect_left(demand_level, d)
        if level != len(demand_level):
            price = price_levels[level]
        else:
            price = price_levels[-1]
        price_day.append(price)

        if "piece-wise" in cost_function_type and level > 0:
            cost += demand_level[0] * price_levels[0]
            cost += (d - demand_level[level - 1]) * price
            cost += sum([(demand_level[i] - demand_level[i - 1]) * price_levels[i] for i in range(1, level)])
        else:
            cost += d * price

    return price_day, cost


def pricing_master_problem(price_levels, demand_table, demand_profile_pre, demand_profile_new, cost_type):
    best_step_size = 1
    demand_profile_updated = demand_profile_new

    # simply compute the total consumption cost and the price
    if demand_profile_pre is None:
        price_day, cost = pricing_cost(demand_profile_new, price_levels, demand_table, cost_type)

    # apply the FW algorithm
    else:
        step_profile = []
        for dp, dn, d_levels in zip(demand_profile_pre, demand_profile_new, demand_table):
            step = 1
            dd = dn - dp
            if dd != 0:
                dl = find_ge(d_levels, dp) if dd > 0 else find_le(d_levels, dp)
                step = (dl - dp) / dd
            step_profile.append(step)

        best_step_size = min(step_profile)
        demand_profile_updated = [dp + (dn - dp) * best_step_size for dp, dn in
                                  zip(demand_profile_pre, demand_profile_new)]
        price_day, cost = pricing_cost(demand_profile_updated, price_levels, demand_table, cost_type)

    return demand_profile_updated, best_step_size, price_day, cost


def iteration():
    # print experiment summary
    print("---------- Summary ----------")
    print("{0} households, {1} tasks per household".format(no_households, no_tasks))
    print("---------- Experiments begin! ----------")

    # 0 - generation household data and the total preferred demand profile
    if new_households:
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods, file_household_area_folder,
                                           no_households, no_tasks, care_f_weight, care_f_max, max_demand_multiplier,
                                           dp_profile, dp_interval, dp_period, dp_optimal, dp_heuristic)
        print("Household data created...")
    else:
        households, area = area_read(file_household_area_folder)
        print("Household data read...")
    area_demand_profile = area[dp_profile][dp_period][0]
    area_demand_max = max(area_demand_profile)

    # 0 - read the model file, solver choice and the pricing table (price levels and the demand table)
    models, solvers, price_levels, demand_table \
        = read_data(file_cp_pre, file_cp_ini, file_pricing_table, area_demand_max, pricing_table_weight)
    solver_choice = solvers[solver_type]
    model_file = models[solver_type][model_type]
    print("Model, solver and pricing data created...")

    # 0 - the prices of the total preferred demand profile
    demand_profile_unchanged, step_size, price_day, cost \
        = pricing_master_problem(price_levels, demand_table, None, area_demand_profile, cost_type)
    print("First day prices calculated...")

    # 0 - initialise trackers
    step_size_history = [step_size]
    cost_history = [cost]
    prices_history = [price_day]
    print("---------- Initialisation done! ----------")

    # 1 - rescheduling and pricing (iteration = k where k > 0)
    k = 1
    total_runtime_heuristic = 0
    total_runtime_optimal = 0
    while True:

        # 1 - reschedule given the prices at iteration k - 1
        prices_pre = prices_history[k - 1]
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
        area[dp_profile][dp_heuristic][k] = [sum(x) for x in
                                             grouper(area_demand_profile_heuristic, no_intervals_periods)]
        area[dp_profile][dp_optimal][k] = [sum(x) for x in grouper(area_demand_profile_optimal, no_intervals_periods)]

        # 1 - pricing


iteration()