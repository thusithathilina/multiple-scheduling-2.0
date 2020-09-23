from minizinc import *
from numpy import sqrt, pi, random
import numpy as np
import random as r
from csv import reader
from more_itertools import grouper
from bisect import bisect_left
from multiple.cfunctions import find_ge, find_le
# from pandas import DataFrame, IndexSlice, date_range, Series, concat
from numpy import genfromtxt
import pickle
from time import strftime, localtime
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
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


def read_data(f_cp_pre, f_cp_ini, f_pricing_table, demand_limit):
    models = dict()
    models["cp"] = dict()
    models["cp"]["pre"] = f_cp_pre
    models["cp"]["init"] = f_cp_ini

    solvers = dict()
    solvers["cp"] = "gecode"

    demand_levels = []
    price_levels = []
    demand_level_scale = demand_limit * pricing_table_weight
    with open(f_pricing_table, 'r') as csvfile:
        csvreader = reader(csvfile, delimiter=',', quotechar='|')

        for row in csvreader:
            pricing_table_row = list(map(float, row))
            price_levels.append(pricing_table_row[0])
            demand_levels.append([round(x * demand_level_scale, -3) for x in pricing_table_row[1:]])
    demand_table = zip(*demand_levels)

    return models, solvers, price_levels, demand_table


def task_generation(num_intervals, num_periods, num_intervals_periods, mode_value, l_demands, p_d_short):
    # generation - demand
    demand = r.choice(l_demands)
    demand = int(demand * 1000)

    # generation - duration
    duration = max(1, int(random.rayleigh(mode_value, 1)[0]))

    # generation - preferred start time
    p_start = max(int(np.random.choice(a=num_periods, size=1, p=p_d_short)[0]) * num_intervals_periods
                  + r.randint(-num_intervals_periods + 1, num_intervals_periods), 0)
    p_start = min(p_start, num_intervals - 1)

    # generation - earliest starting time
    # e_start = r.randint(-duration + 1, p_start)
    e_start = 0

    # generation - latest finish time
    l_finish = r.randint(p_start + duration, num_intervals - 1 + duration)

    # generation - care factor
    care_f = int(r.choice([i for i in range(care_f_max + 1)]))

    return demand, duration, p_start, e_start, l_finish, care_f


def household_generation(num_intervals, num_periods, num_intervals_periods, num_tasks, p_d):
    p_d_short = [int(p) for p in p_d[0]]
    sum_t = sum(p_d_short)
    p_d_short = [p / sum_t for p in p_d_short]

    l_demands = genfromtxt('inputs/demands_list.csv', delimiter=',', dtype="float")

    # I meant mean value is 40 minutes
    mean_value = 40.0 / (24.0 * 60.0 / num_intervals)
    mode_value = sqrt(2 / pi) * mean_value

    # task details
    preferred_starts = []
    earliest_starts = []
    latest_ends = []
    durations = []
    demands = []
    care_factors = []
    aggregated_loads = [0] * num_intervals

    # tasks in the household
    for counter_j in range(num_tasks):
        demand, duration, p_start, e_start, l_finish, care_f \
            = task_generation(num_intervals, num_periods, num_intervals_periods, mode_value, l_demands, p_d_short)
        demands.append(demand)
        durations.append(duration)
        preferred_starts.append(p_start)
        earliest_starts.append(e_start)
        latest_ends.append(l_finish)
        care_factors.append(care_f)
        # add this task demand to the household demand
        for d in range(duration):
            aggregated_loads[(p_start + d) % num_intervals] += demand
    # set the household demand limit
    maximum_demand = max(demands) * max_demand_multiplier

    # precedence among tasks
    precedors = dict()
    no_precedences = 0
    succ_delays = dict()

    def retrieve_precedors(list0):
        list3 = []
        for l in list0:
            if l in precedors:
                list2 = precedors[l]
                retrieved_list = retrieve_precedors(list2)
                # todo - add the one with the latest finish? earliest start time?
                list3.extend(retrieved_list)
            else:
                list3.append(l)
        return list3

    def add_precedes(t, prev, delay):
        if t not in precedors:
            precedors[t] = [prev]
            succ_delays[t] = [delay]
        else:
            precedors[t].append(prev)
            succ_delays[t].append(delay)

    for t in range(1, num_tasks):
        # if r.choice([True, False]):
        if True:
            previous_tasks = list(range(t))
            # previous_tasks.reverse()
            r.shuffle(previous_tasks)
            for prev in previous_tasks:
                if preferred_starts[prev] + durations[prev] - 1 < preferred_starts[t] \
                        and earliest_starts[prev] + durations[prev] < latest_ends[t] - durations[t] + 1:

                    if prev not in precedors:
                        # feasible delay
                        succeding_delay = num_intervals - 1
                        add_precedes(t, prev, succeding_delay)
                        no_precedences += 1

                        break
                    else:
                        # find all precedors of this previous task
                        precs_prev = retrieve_precedors([prev])
                        precs_prev.append(prev)

                        precs_prev_duration = sum([durations[x] for x in precs_prev])
                        latest_pstart = preferred_starts[precs_prev[0]]
                        latest_estart = earliest_starts[precs_prev[0]]

                        if latest_pstart + precs_prev_duration - 1 < preferred_starts[t] \
                                and latest_estart + precs_prev_duration < latest_ends[t] - durations[t] + 1:
                            succeding_delay = num_intervals - 1
                            add_precedes(t, prev, succeding_delay)
                            no_precedences += 1
                            break

    # print(" --- Household made ---")

    return preferred_starts, earliest_starts, latest_ends, durations, demands, care_factors, \
           no_precedences, precedors, succ_delays, maximum_demand, aggregated_loads


def area_generation(num_intervals, num_periods, num_intervals_periods, data_folder):
    probability = genfromtxt('inputs/probability.csv', delimiter=',', dtype="float")

    households = dict()
    area_demand_profile = [0] * num_intervals

    for h in range(no_households):
        preferred_starts, earliest_starts, latest_ends, durations, demands, care_factors, \
        num_precedences, precedors, succ_delays, max_demand, household_profile \
            = household_generation(num_intervals, num_periods, num_intervals_periods, no_tasks, probability)

        household_key = h
        households[household_key] = dict()

        households[household_key]["demands"] = demands
        households[household_key]["durs"] = durations
        households[household_key]["ests"] = earliest_starts
        households[household_key]["lfts"] = latest_ends
        households[household_key]["psts"] = preferred_starts
        households[household_key]["cfs"] = [cf * care_f_weight for cf in care_factors]
        households[household_key]["precs"] = precedors
        households[household_key]["succ_delays"] = succ_delays
        households[household_key]["no_prec"] = num_precedences

        households[household_key]["profile", "preferred"] = household_profile
        households[household_key]["max", "preferred"] = max(household_profile)
        households[household_key]["max", "limit"] = max_demand

        area_demand_profile = [x + y for x, y in zip(household_profile, area_demand_profile)]

    # area demand profile
    area = dict()
    area[dp_profile] = dict()
    area[dp_profile][dp_interval] = dict()
    area[dp_profile][dp_interval][0] = area_demand_profile
    area[dp_profile][dp_period] = dict()
    area[dp_profile][dp_period][0] = [sum(x) for x in grouper(area_demand_profile, num_intervals_periods)]
    area[dp_profile][dp_optimal] = dict()
    area[dp_profile][dp_optimal][0] = area_demand_profile
    area[dp_profile][dp_heuristic] = dict()
    area[dp_profile][dp_heuristic][0] = area_demand_profile

    # write household data and area data into files\
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    with open(data_folder + "households" + '.pkl', 'wb+') as f:
        pickle.dump(households, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    with open(data_folder + "area" + '.pkl', 'wb+') as f:
        pickle.dump(area, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    return households, area


def area_read(data_folder):
    with open(data_folder + "households" + '.pkl', 'rb') as f:
        households = pickle.load(f)
    f.close()

    with open(data_folder + "area" + '.pkl', 'rb') as f:
        area = pickle.load(f)
    f.close()

    return households, area


def data_preprocessing(num_intervals, num_tasks, prices_day, earliest_starts, latest_ends, durations,
                       preferred_starts, care_factors, demands, cf_weight):
    max_demand = max(demands)
    max_duration = max(durations)
    run_costs = []
    big_cost = (num_intervals * care_f_max * max_demand + max_demand * max_duration)
    for i in range(num_tasks):
        demand = demands[i]
        pstart = preferred_starts[i]
        estart = earliest_starts[i]
        lfinish = latest_ends[i]
        duration = durations[i]
        cfactor = care_factors[i]

        run_cost_task = []
        for t in range(num_intervals):
            if estart <= t <= lfinish - duration + 1:
                rc = abs(t - pstart) * cfactor * cf_weight
                rc += sum([prices_day[j % num_intervals] for j in range(t, t + duration)]) * demand
            else:
                rc = big_cost
            run_cost_task.append(int(rc))
        run_costs.append(run_cost_task)
    return run_costs


def household_heuristic_solving(num_intervals, num_tasks, durations, demands, predecessors, successors,
                                succ_delays, max_demand, run_costs, precedents, latest_ends):
    start_time = timeit.default_timer()

    actual_starts = []
    household_profile = [0] * num_intervals
    obj = 0
    max_duration = max(durations)
    big_cost = (num_intervals * care_f_max * max_demand + max_demand * max_duration)

    def retrieve_successors_or_precedents(list0, prec_or_succ_list1, succ_prec_list2):
        list_r = []
        for l in list0:
            if l in prec_or_succ_list1:
                succ_or_prec_indices = [i2 for i2, k in enumerate(prec_or_succ_list1) if k == l]
                succ_or_prec = [succ_prec_list2[i2] for i2 in succ_or_prec_indices]
                succ_succ_or_prec_prec = retrieve_successors_or_precedents(succ_or_prec, prec_or_succ_list1, succ_prec_list2)
                list_r.extend(succ_succ_or_prec_prec)
            else:
                list_r.append(l)
        return list_r

    def check_if_successors_or_precedents_exist(checked_task_id, prec_or_succ1, succ_or_prec2):
        succs_succs_or_precs_precs = []
        if checked_task_id in prec_or_succ1:
            indices = [i2 for i2, k in enumerate(prec_or_succ1) if k == checked_task_id]
            succs_or_precs = [succ_or_prec2[i2] for i2 in indices]
            succs_succs_or_precs_precs = retrieve_successors_or_precedents(succs_or_precs, prec_or_succ1, succ_or_prec2)

        return succs_succs_or_precs_precs

    for task_id in range(num_tasks):
        demand = demands[task_id]
        duration = durations[task_id]
        task_costs = run_costs[task_id]

        # if i has successors
        tasks_successors = check_if_successors_or_precedents_exist(task_id, predecessors, successors)
        earliest_suc_lstart = num_intervals - 1
        if bool(tasks_successors):
            suc_durations = [durations[i2] for i2 in tasks_successors]
            suc_lends = [latest_ends[i2] for i2 in tasks_successors]
            total_suc_duration = sum(suc_durations)
            earliest_suc_lstart = min(suc_lends) - total_suc_duration + 1
            # suc_lstarts = [lend - dur + 1 for lend, dur in zip(suc_lends, suc_durations)]
            # earliest_suc_lstart = min(suc_lstarts)

        # if i has precedents
        tasks_precedents = check_if_successors_or_precedents_exist(task_id, successors, predecessors)
        latest_pre_finish = 0
        latest_pre_finish_w_delay = num_intervals - 1
        if bool(tasks_precedents):
            prec_durations = [durations[i2] for i2 in tasks_precedents]
            prec_astarts = [actual_starts[i2] for i2 in tasks_precedents]
            succ_delay = succ_delays[task_id]
            latest_pre_finish = max([astart + dur - 1 for astart, dur in zip(prec_durations, prec_astarts)])
            latest_pre_finish_w_delay = latest_pre_finish + succ_delay[0]

        # search for all feasible intervals
        feasible_intervals = []
        for j in range(num_intervals):
            if task_costs[j] < big_cost and j + duration - 1 < earliest_suc_lstart \
                    and latest_pre_finish < j < latest_pre_finish_w_delay:
                feasible_intervals.append(j)

        if not bool(feasible_intervals):
            print("error")

        feasible_min_cost = min([task_costs[f] for f in feasible_intervals])
        cheapest_intervals = [f for f in feasible_intervals if task_costs[f] == feasible_min_cost]
        a_start = r.choice(cheapest_intervals)

        # check max demand constraint
        max_demand_starts = dict()
        temp_profile = household_profile[:]
        try:
            for d in range(a_start, a_start + duration):
                temp_profile[d % num_intervals] += demand
        except IndexError:
            print("error")
        temp_max_demand = max(temp_profile)
        while temp_max_demand > max_demand and len(feasible_intervals) > 1:

            max_demand_starts[a_start] = temp_max_demand
            feasible_intervals.remove(a_start)

            feasible_min_cost = min([run_costs[task_id][f] for f in feasible_intervals])
            feasible_min_cost_indices = [k for k, x in enumerate(run_costs[task_id]) if x == feasible_min_cost]
            a_start = r.choice(feasible_min_cost_indices)

            temp_profile = household_profile[:]
            for d in range(a_start, a_start + duration):
                temp_profile[d] += demand
            temp_max_demand = max(temp_profile)

        if len(feasible_intervals) == 0 and not max_demand_starts:
            a_start = min(max_demand_starts, key=max_demand_starts.get)

        actual_starts.append(a_start)
        for d in range(a_start, a_start + duration):
            household_profile[d % num_intervals] += demand
        obj += run_costs[task_id][a_start]

    elapsed = timeit.default_timer() - start_time

    return actual_starts, household_profile, obj, elapsed


def household_optimal_solving \
                (num_intervals, num_tasks, prices_day, preferred_starts, earliest_starts, latest_ends, durations,
                 demands, care_factors, no_precedences, predecessors, successors, succ_delays, max_demand, model_file,
                 solver_choice, model_type, solver_type, run_costs, search, cf_weight):
    # problem model
    model = Model(model_file)
    gecode = Solver.lookup(solver_choice)
    model.add_string("solve ")
    if "gecode" in solver_choice:
        model.add_string(":: {} ".format(search))
    model.add_string("minimize obj;")

    ins = Instance(gecode, model)
    ins["num_intervals"] = num_intervals
    ins["num_tasks"] = num_tasks
    ins["durations"] = durations
    ins["demands"] = demands
    ins["num_precedences"] = no_precedences
    ins["predecessors"] = [p + 1 for p in predecessors]
    ins["successors"] = [s + 1 for s in successors]
    ins["prec_delays"] = succ_delays
    ins["max_demand"] = max_demand

    if "ini" in model_type.lower():
        ins["prices"] = prices_day
        ins["preferred_starts"] = [ps + 1 for ps in preferred_starts]
        ins["earliest_starts"] = [es + 1 for es in earliest_starts]
        ins["latest_ends"] = [le + 1 for le in latest_ends]
        ins["care_factors"] = [cf * cf_weight for cf in care_factors]
    else:
        ins["run_costs"] = run_costs

    # solve problem model
    result = ins.solve()

    # process problem solution
    obj = result.objective
    solution = result.solution.actual_starts
    if "cp" in solver_type:
        actual_starts = [int(a) - 1 for a in solution]
    else:  # "mip" in solver_type:
        actual_starts = [sum([i * int(v) for i, v in enumerate(row)]) for row in solution]
    time = result.statistics["time"].total_seconds()

    optimal_demand_profile = [0] * num_intervals
    for demand, duration, a_start, i in zip(demands, durations, actual_starts, range(num_tasks)):
        for t in range(a_start, a_start + duration):
            optimal_demand_profile[t % num_intervals] += demand

    return optimal_demand_profile, actual_starts, obj, time


def household_scheduling_subproblem \
                (num_intervals, num_tasks, num_periods, num_intervals_periods, household, prices_day, model_file,
                 m_type, s_type, solver_choice):
    # pricing data
    if np.isnan(prices_day[-1]) or len(prices_day) == num_periods:
        prices_day = [int(p) for p in prices_day[:num_periods] for _ in range(num_intervals_periods)]
    else:
        prices_day = [int(p) for p in prices_day]

    # household data
    demands = household["demands"]
    durations = household["durs"]
    earliest_starts = household["ests"]
    latest_ends = household["lfts"]
    preferred_starts = household["psts"]
    care_factors = household["cfs"]
    precedents = [x[0] for x in list(household["precs"].values())]
    successors = list(household["precs"].keys())
    succ_delays = household["succ_delays"] # need to change this format when sending it to the solver
    no_precedences = household["no_prec"]
    max_demand = household["max", "limit"]

    # data preprocessing
    run_costs = data_preprocessing(no_intervals, no_tasks, prices_day, earliest_starts, latest_ends, durations,
                                   preferred_starts, care_factors, demands, care_f_weight)

    # heuristic algorithm -- optimistic greedy search algorithm (OGSA)
    optimistic_starts, optimistic_profile, optimistic_obj, optimistic_runtime \
        = household_heuristic_solving(num_intervals, num_tasks, durations, demands, precedents, successors,
                                      succ_delays, max_demand, run_costs[:], household["precs"], latest_ends)

    # optimisation solver -- constraint programming (CP)
    succ_delays = [x[0] for x in list(household["succ_delays"].values())]
    search = "int_search(actual_starts, {}, {}, complete)".format(var_selection, val_choice)
    optimal_profile, optimal_starts, optimal_obj, optimal_runtime \
        = household_optimal_solving(num_intervals, num_tasks, prices_day, preferred_starts, earliest_starts,
                                    latest_ends, durations, demands, care_factors, no_precedences, precedents,
                                    successors, succ_delays, max_demand, model_file, solver_choice, m_type, s_type,
                                    run_costs[:], search, care_f_weight)

    return optimistic_starts, optimistic_profile, optimistic_obj, optimistic_runtime, \
        optimal_starts, optimal_profile, optimal_obj, optimal_runtime


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
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods, file_household_area_folder)
        print("Household data created...")
    else:
        households, area = area_read(file_household_area_folder)
        print("Household data read...")
    area_demand_profile = area[dp_profile][dp_period][0]
    area_demand_max = max(area_demand_profile)

    # 0 - read the model file, solver choice and the pricing table (price levels and the demand table)
    models, solvers, price_levels, demand_table \
        = read_data(file_cp_pre, file_cp_ini, file_pricing_table, area_demand_max)
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
                = household_scheduling_subproblem(no_intervals, no_tasks, no_periods, no_intervals_periods, household,
                                                  prices_pre, model_file,
                                                  model_type, solver_type, solver_choice)

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

# def optimised_profile(solutions, s_type, demands, durations):
#     solutions = list(solutions.values())[0]
#
#     if s_type.lower() == "mip":
#         actual_starts = [sum([i * int(v) for i, v in enumerate(row)]) for row in solutions]
#     else:
#         # need to change the index back to starting from 0!!!!!
#         actual_starts = [int(a) - 1 for a in solutions]
#
#     optimised_demand_profile = [0] * no_intervals
#     for demand, duration, a_start, i in zip(demands, durations, actual_starts, range(no_tasks)):
#         for t in range(a_start, a_start + duration):
#             optimised_demand_profile[t] += demand
#
#     return optimised_demand_profile, actual_starts

# if i in predecessors:
#     task_successors_indices = [i2 for i2, k in enumerate(predecessors) if k == i]
#     task_successors = [successors[i2] for i2 in task_successors_indices]
#     # find the successors of successors
#     task_successors_successors = retrieve_successors_or_precedents(task_successors, predecessors)
#     suc_durations = [durations[i2] for i2 in task_successors_successors]
#     suc_lends = [latest_ends[i2] for i2 in task_successors_successors]
#     suc_lstarts = [lend - dur + 1 for lend, dur in zip(suc_lends, suc_durations)]
#     earliest_suc_lstart = min(suc_lstarts)

# if i has a predecessor

# if i in successors:
#     task_prec_indics = [i2 for i2, k in enumerate(successors) if k == i]
#     task_precedents = [predecessors[i2] for i2 in task_prec_indics]
#     # find the precedents of precedents
#     task_precedents_precedents = retrieve_successors_or_precedents(task_precedents, successors)
#
#     prec_durations = [durations[i2] for i2 in task_precedents_precedents]
#     prec_astarts = [actual_starts[i2] for i2 in task_precedents_precedents]
#     pre_delay = [succ_delays[i2] for i2 in task_precedents_precedents]

# index = successors.index(i)
# pre_id = predecessors[index]
# pre_delay = succ_delays[index]
# pre_astart = actual_starts[pre_id]
# pre_duration = durations[pre_id]
# pre_finish_w_delay = pre_astart + pre_duration - 1 + pre_delay
