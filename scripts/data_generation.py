import pickle
from numpy import genfromtxt
from csv import reader
import random as r
import numpy as np
from numpy import sqrt, pi, random
import os
from more_itertools import grouper
from scripts.input_parameter import *
from scripts.cfunctions import average


def read_data(f_cp_pre, f_cp_ini, f_pricing_table, demand_level_scale, zero_digit):
    models = dict()
    models["cp"] = dict()
    models["cp"]["pre"] = f_cp_pre
    models["cp"]["init"] = f_cp_ini

    solvers = dict()
    solvers["cp"] = "gecode"

    pricing_table = dict()
    pricing_table[k0_price_levels] = []
    pricing_table[k0_demand_table] = dict()
    with open(f_pricing_table, 'r') as csvfile:
        csvreader = reader(csvfile, delimiter=',', quotechar='|')

        for i_row, row in enumerate(csvreader):
            # a row - the price and the demands of all periods at one level.
            pricing_table_row = list(map(float, row))
            price = pricing_table_row[0]
            pricing_table[k0_price_levels].append(price)
            # a col - the demand at one level of a period
            for i_col, col in enumerate(pricing_table_row[1:]):
                if i_col not in pricing_table[k0_demand_table]:
                    pricing_table[k0_demand_table][i_col] = dict()
                pricing_table[k0_demand_table][i_col][i_row] = round(col * demand_level_scale, -zero_digit)

    return models, solvers, pricing_table


def task_generation(num_intervals, num_periods, num_intervals_periods, mode_value, l_demands, p_d_short, cf_max):
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
    care_f = int(r.choice([i for i in range(cf_max + 1)]))

    return demand, duration, p_start, e_start, l_finish, care_f


def household_generation(num_intervals, num_periods, num_intervals_periods, num_tasks, p_d,
                         max_demand_multiplier, cf_max, f_demand_list):
    p_d_short = [int(p) for p in p_d[0]]
    sum_t = sum(p_d_short)
    p_d_short = [p / sum_t for p in p_d_short]

    l_demands = genfromtxt(f_demand_list, delimiter=',', dtype="float")

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
            = task_generation(num_intervals, num_periods, num_intervals_periods,
                              mode_value, l_demands, p_d_short, cf_max)
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

    def retrieve_precedes(list0):
        list3 = []
        for l in list0:
            if l in precedors:
                list2 = precedors[l]
                retrieved_list = retrieve_precedes(list2)
                list3.extend(retrieved_list)
            else:
                list3.append(l)
        return list3

    def add_precedes(task, previous, delay):
        if task not in precedors:
            precedors[task] = [previous]
            succ_delays[task] = [delay]
        else:
            precedors[task].append(previous)
            succ_delays[task].append(delay)

    for t in range(num_tasks, num_tasks):
        if r.choice([True, False]):
        # if True:
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
                        precs_prev = retrieve_precedes([prev])
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


def area_generation(num_intervals, num_periods, num_intervals_periods, data_folder,
                    num_households, num_tasks, cf_weight, cf_max, max_d_multiplier,
                    f_probability, f_demand_list, algorithms_labels):
    probability = genfromtxt(f_probability, delimiter=',', dtype="float")

    households = dict()
    area_demand_profile = [0] * num_intervals

    for h in range(num_households):
        preferred_starts, earliest_starts, latest_ends, durations, demands, care_factors, \
        num_precedences, precedors, succ_delays, max_demand, household_profile \
            = household_generation(num_intervals, num_periods, num_intervals_periods, num_tasks,
                                   probability, max_d_multiplier, cf_max, f_demand_list)

        household_key = h
        households[household_key] = dict()

        households[household_key]["demands"] = demands
        households[household_key]["durs"] = durations
        households[household_key]["ests"] = earliest_starts
        households[household_key]["lfts"] = latest_ends
        households[household_key]["psts"] = preferred_starts
        households[household_key]["cfs"] = [cf * cf_weight for cf in care_factors]
        households[household_key]["precs"] = precedors
        households[household_key]["succ_delays"] = succ_delays
        households[household_key]["no_prec"] = num_precedences

        households[household_key]["profile", "preferred"] = household_profile
        households[household_key]["max", "preferred"] = max(household_profile)
        households[household_key]["max", "limit"] = max_demand

        households[household_key][k0_starts] = dict()

        for k in algorithms_labels.keys():
            households[household_key][k0_starts][k] = dict()
            # households[household_key][k0_starts][k1_optimal_scheduling] = dict()

            households[household_key][k0_starts][k][0] = preferred_starts
            # households[household_key][k0_starts][k1_optimal_scheduling][0] = preferred_starts

        area_demand_profile = [x + y for x, y in zip(household_profile, area_demand_profile)]

    area = dict()
    area[k0_summary] = dict()
    area[k0_summary][k1_tasks_no] = num_tasks
    area[k0_summary][k1_households_no] = num_households
    area[k0_summary][k1_penalty_weight] = num_households

    def initialise_area_trackers(k0_key, k1_key):
        if k0_key not in area:
            area[k0_key] = dict()
        area[k0_key][k1_key] = dict()

    # initialise trackers
    area_demand_profile_pricing = [sum(x) for x in grouper(area_demand_profile, num_intervals_periods)]
    # track four types of demand profiles, prices, objective values, costs, penalties, max demands and PARs
    k0_keys = [k0_demand, k0_prices, k0_obj, k0_cost, k0_penalty, k0_time, k0_step]
    # k1_keys = [k1_optimal_scheduling, k1_heuristic_scheduling, k1_optimal_fw, k1_heuristic_fw]
    area[k0_demand_max] = dict()
    area[k0_demand_total] = dict()
    area[k0_par] = dict()
    for k0 in k0_keys:
        for alg in algorithms_labels.values():
            for k1 in alg.values():
                initialise_area_trackers(k0, k1)
                # initial values for four kinds of demand profiles, max demands, PARs and the penalty
                if k0 == k0_demand:
                    area[k0][k1][0] = area_demand_profile_pricing
                    max_demand = max(area_demand_profile_pricing)
                    area[k0_demand_max][k1] = dict()
                    area[k0_demand_total][k1] = dict()
                    area[k0_par][k1] = dict()
                    area[k0_demand_max][k1][0] = max_demand
                    area[k0_demand_total][k1][0] = sum(area_demand_profile_pricing)
                    area[k0_par][k1][0] = average(area_demand_profile_pricing) / max_demand

                if k0 == k0_penalty:
                    area[k0_penalty][k1][0] = 0

                if k0 == k0_step:
                    if "fw" in k1:
                        initialise_area_trackers(k0, k1)

    # write household data and area data into files
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
