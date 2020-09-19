from minizinc import *
from numpy import sqrt, pi, random
import numpy as np
import random as r

from pandas import DataFrame, IndexSlice, date_range, Series, concat
from numpy import genfromtxt
from time import strftime, localtime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import timeit


no_intervals = 144
no_periods = 48
no_intervals_periods = int(no_intervals / no_periods)
no_households = 1000
no_tasks = 5
max_demand_multiplier = no_tasks
care_f_max = 10
care_f_weight = 800

var_choices = ""
const_choices = ""


def read_data():
    prices = genfromtxt('inputs/prices.csv', delimiter=',', dtype="float").astype(int) # in cents

    models = dict()
    models["CP"] = dict()
    models["CP"]["pre"] = 'models/Household-cp-pre.mzn'
    models["CP"]["init"] = 'models/Household-cp.mzn'

    solvers = dict()
    solvers["CP"] = "Gecode"

    return prices, models, solvers


def task_generation(mode_value, l_demands, p_d_short):

    # job consumption per hour
    demand = r.choice(l_demands)
    demand = int(demand * 1000)

    # job duration
    duration = max(1, int(random.rayleigh(mode_value, 1)[0]))
    # while duration == 0:
    #     duration = int(random.rayleigh(mode_value, 1)[0])

    # job preferred start time
    p_start = no_intervals + 1
    while p_start + duration - 1 >= no_intervals - 1 or p_start < 0:
        middle_point = int(np.random.choice(a=no_periods, size=1, p=p_d_short)[0]
                           * no_intervals_periods
                           + np.random.random_integers(low=-2, high=2))
        p_start = middle_point - int(duration / 2)

    # job earliest starting time
    e_start = 0
    # e_start = r.randint(0, max(p_start - 1, 0))

    # job latest finish time
    l_finish = no_intervals - 1
    # l_finish = r.randint(p_start + duration, min(no_intervals - 1, p_start + duration))

    # job care factor
    care_f = int(r.choice([i for i in range(care_f_max + 1)]))

    return demand, duration, p_start, e_start, l_finish, care_f


def household_generation(p_d):
    p_d_short = [int(p) for p in p_d[0]]
    sum_t = sum(p_d_short)
    p_d_short = [p / sum_t for p in p_d_short]

    l_demands = genfromtxt('inputs/demands_list.csv', delimiter=',', dtype="float")

    # I meant mean value is 40 minutes
    mean_value = 40.0 / (24.0 * 60.0 / no_intervals)
    mode_value = sqrt(2 / pi) * mean_value

    # task details
    preferred_starts = []
    earliest_starts = []
    latest_ends = []
    durations = []
    demands = []
    care_factors = []
    predecessors = []
    successors = []
    prec_delays = []
    aggregated_loads = [0] * no_intervals

    # tasks in the household
    for counter_j in range(no_tasks):

        demand, duration, p_start, e_start, l_finish, care_f = task_generation(mode_value, l_demands, p_d_short)
        demands.append(demand)
        durations.append(duration)
        preferred_starts.append(p_start)
        earliest_starts.append(e_start)
        latest_ends.append(l_finish)
        care_factors.append(care_f)

        # decide if this task has a predecessing task
        if r.choice([True, False]) and counter_j > 0:

            # task predecessor
            id_predecessor_set = [i for i in range(counter_j)]
            id_predecessor = r.choice(id_predecessor_set)

            while preferred_starts[id_predecessor] + durations[id_predecessor] - 1 >= preferred_starts[counter_j] \
                    and len(id_predecessor_set) > 0:
                id_predecessor_set.remove(id_predecessor)
                if len(id_predecessor_set) > 0:
                    id_predecessor = r.choice(id_predecessor_set)

            if len(id_predecessor_set) > 0:
                predecessors.append(int(id_predecessor))
                successors.append(counter_j)

                # predecessing delay
                delay = 0
                if not durations[id_predecessor] + duration - 1 == no_intervals - 1:
                    delay = r.choice([i for i in range(no_intervals + 1 - duration - durations[id_predecessor])])
                prec_delays.append(int(delay))

        # add this task demand to the household demand
        for d in range(duration):
            aggregated_loads[p_start + d] += demand

    # set the household demand limit
    no_precedences = len(predecessors)
    maximum_demand = max(demands) * max_demand_multiplier

    # print(" --- Household made ---")

    return preferred_starts, earliest_starts, latest_ends, durations, demands, care_factors, \
           no_precedences, predecessors, successors, prec_delays, maximum_demand, aggregated_loads


def area_generation():
    probability_demand_profile = genfromtxt('inputs/probability.csv', delimiter=',', dtype="float")

    households = dict()

    for h in range(no_households):

        preferred_starts, earliest_starts, latest_ends, durations, demands, care_factors, \
        num_precedences, predecessors, successors, prec_delays, max_demand, household_profile \
            = household_generation(probability_demand_profile)

        household_key = h
        households[household_key] = dict()

        households[household_key]["demands"] = demands
        households[household_key]["durs"] = durations
        households[household_key]["ests"] = earliest_starts
        households[household_key]["lfts"] = latest_ends
        households[household_key]["psts"] = preferred_starts
        households[household_key]["cfs"] = [cf * care_f_weight for cf in care_factors]
        households[household_key]["precs"] = predecessors + [-1] * (no_tasks - num_precedences)
        households[household_key]["succs"] = successors + [-1] * (no_tasks - num_precedences)
        households[household_key]["prec_delays"] = prec_delays + [-1] * (no_tasks - num_precedences)
        households[household_key]["no_prec"] = num_precedences

        households[household_key]["profile", "preferred"] = household_profile
        households[household_key]["max", "preferred"] = max(household_profile)
        households[household_key]["max","limit"] = max_demand

    return households


def data_preprocessing(prices_input, earliest_starts, latest_ends, durations, preferred_starts, care_factors, demands,
                       care_f_weight):
    run_costs = []
    for i in range(no_tasks):
        run_cost_task = []
        for t in range(no_intervals):
            if earliest_starts[i] <= t <= latest_ends[i] - durations[i] + 1:
                rc = abs(t - preferred_starts[i]) * care_factors[i] * care_f_weight \
                     + sum([prices_input[t2] for t2 in range(t, t + durations[i])]) * demands[i]
            else:
                rc = (no_intervals * care_f_max * care_f_weight + max(prices_input) * max(demands) * max(durations))
            run_cost_task.append(int(rc))
        run_costs.append(run_cost_task)
    return run_costs


def optimistic_search(num_intervals, num_tasks, durations, demands, predecessors, successors,
                      prec_delays, max_demand, run_costs, preferred_starts, latest_ends):
    start_time = timeit.default_timer()

    actual_starts = []
    household_profile = [0] * num_intervals
    obj = 0

    for i in range(num_tasks):

        task_demand = demands[i]
        task_duration = durations[i]

        max_cost = max(run_costs[i])
        feasible_indices = [t for t, x in enumerate(run_costs[i]) if x < max_cost]

        if i in successors:
            pos_in_successors = successors.index(i)
            pre_id = predecessors[pos_in_successors]
            pre_delay = prec_delays[pos_in_successors]
            this_aestart = actual_starts[pre_id] + durations[pre_id]
            feasible_indices = [f for f in feasible_indices if this_aestart <= f <= this_aestart + pre_delay]

        if i in predecessors:
            indices_in_precs = [i for i, k in enumerate(predecessors) if k == i]
            for iip in indices_in_precs:
                suc_id = successors[iip]
                suc_duration = durations[suc_id]
                suc_lend = latest_ends[suc_id]
                feasible_indices = [f for f in feasible_indices if f + task_duration + suc_duration - 1 <= suc_lend]

        # if not this_successors == []:
        #     last_suc_lend = latest_ends[max(this_successors)]
        #     sum_successors_durations = sum([durations[t] for t in this_successors])
        #     for j in feasible_indices:
        #         if not (j + task_duration + sum_successors_durations - 1 <= last_suc_lend):
        #             feasible_indices.remove(j)

        # last_prec_delay = -1
        # this_predecessors = []
        # while this_id in successors:

        # if not this_predecessors == []:
        #     first_prec_astart = solutions[min(this_predecessors)]
        #     sum_prec_durations = sum([durations[t] for t in this_predecessors])
        #     this_estart = first_prec_astart + sum_prec_durations
        #     for j in feasible_indices:
        #         if not (this_estart <= j <= this_estart + last_prec_delay):
        #             feasible_indices.remove(j)

        if feasible_indices is []:
            print("error")

        feasible_min_cost = min([run_costs[i][f] for f in feasible_indices])
        feasible_min_cost_indices = [f for f in feasible_indices if run_costs[i][f] == feasible_min_cost]
        a_start = r.choice(feasible_min_cost_indices)

        # check max demand constraint
        max_demand_starts = dict()
        temp_profile = household_profile[:]
        try:
            for d in range(a_start, a_start + task_duration):
                temp_profile[d] += task_demand
        except:
            print("error")
        temp_max_demand = max(temp_profile)
        while temp_max_demand > max_demand and len(feasible_indices) > 1:

            max_demand_starts[a_start] = temp_max_demand
            feasible_indices.remove(a_start)

            feasible_min_cost = min([run_costs[i][f] for f in feasible_indices])
            feasible_min_cost_indices = [k for k, x in enumerate(run_costs[i]) if x == feasible_min_cost]
            a_start = r.choice(feasible_min_cost_indices)

            temp_profile = household_profile[:]
            for d in range(a_start, a_start + task_duration):
                temp_profile[d] += task_demand
            temp_max_demand = max(temp_profile)

        if len(feasible_indices) == 0 and not max_demand_starts:
            a_start = min(max_demand_starts, key=max_demand_starts.get)

        actual_starts.append(a_start)
        for d in range(a_start, a_start + task_duration):
            household_profile[d] += task_demand
        obj += run_costs[i][a_start]

    elapsed = timeit.default_timer() - start_time

    return actual_starts, household_profile, obj, elapsed


def optimal_solving(no_intervals, prices_day, preferred_starts,
                    no_tasks, earliest_starts, latest_ends, durations,
                    demands, care_factors, no_precedences, predecessors,
                    successors, prec_delays, max_demand,
                    model_file, solver_choice, model_type, solver_type,
                    run_costs, search, care_f_weight):

    sing_dsp = Instance([model_file])
    sing_dsp["num_intervals"] = no_intervals
    sing_dsp["num_tasks"] = no_tasks
    sing_dsp["durations"] = durations
    sing_dsp["demands"] = demands
    sing_dsp["num_precedences"] = no_precedences
    sing_dsp["predecessors"] = [p + 1 for p in predecessors]
    sing_dsp["successors"] = [s + 1 for s in successors]
    sing_dsp["prec_delays"] = prec_delays
    sing_dsp["max_demand"] = max_demand

    if "ini" in model_type.lower():
        sing_dsp["prices"] = prices_day
        sing_dsp["preferred_starts"] = [ps + 1 for ps in preferred_starts]
        sing_dsp["earliest_starts"] = [es + 1 for es in earliest_starts]
        sing_dsp["latest_ends"] = [le + 1 for le in latest_ends]
        sing_dsp["care_factors"] = [cf * care_f_weight for cf in care_factors]
    else:
        sing_dsp["run_costs"] = run_costs

    sing_dsp.add_to_model("solve ")
    if solver_choice == "Gecode":
        sing_dsp.add_to_model(":: {} ".format(search))
    sing_dsp.add_to_model("minimize obj;")

    solver = load_solver(solver_choice.lower())

    # if solver_choice == "Chuffed" or "ORtools":
    #     result = solver.solve(sing_dsp, tags=[s_type], free_search=True)
    # else:
    #     result = solver.solve(sing_dsp, tags=[s_type])
    #
    result = solver.solve(sing_dsp, tags=[solver_type])
    # result = solver.solve(sing_dsp)

    # print(solver_choice)

    return result


def optimised_profile(solutions, s_type, demands, durations):
    solutions = list(solutions.values())[0]

    if s_type.lower() == "mip":
        actual_starts = [sum([i * int(v) for i, v in enumerate(row)]) for row in solutions]
    else:
        # need to change the index back to starting from 0!!!!!
        actual_starts = [int(a) - 1 for a in solutions]

    optimised_demand_profile = [0] * no_intervals
    for demand, duration, a_start, i in zip(demands, durations, actual_starts, range(no_tasks)):
        for t in range(a_start, a_start + duration):
            optimised_demand_profile[t] += demand

    return optimised_demand_profile, actual_starts


def household_scheduling_subproblem(household, prices_day, model_file, model_type, solver_type, solver_choice):
    if np.isnan(prices_day[-1]) or len(prices_day) == no_periods:
        prices_day = [int(p) for p in prices_day[:no_periods] for _ in range(no_intervals_periods)]
    else:
        prices_day = [int(p) for p in prices_day]

    demands = household["demands"]
    durations = household["durs"]
    earliest_starts = household["ests"]
    latest_ends = household["lfts"]
    preferred_starts = household["psts"]
    care_factors = household["cfs"]
    predecessors = household["precs"]
    successors = household["succs"]
    prec_delays = household["prec_delays"]
    no_precedences = household["no_prec"]
    max_demand = household["max","limit"]
    run_costs = data_preprocessing(prices_day, earliest_starts, latest_ends, durations, preferred_starts,
                                   care_factors, demands, care_f_weight)

    # heuristics
    sactual_starts_ogsa, optimistic_d_profile, obj_ogsa, run_time_ogsa \
        = optimistic_search(no_intervals, no_tasks, durations, demands, predecessors, successors,
                            prec_delays, max_demand, run_costs[:], preferred_starts, latest_ends)

    # solver
    obj = 0
    sol = [0] * no_intervals
    search = "int_search(actual_starts, {}, {}, complete)".format(var_choices, const_choices)
    solver_results = optimal_solving(no_intervals, prices_day, preferred_starts,
                                     no_tasks, earliest_starts, latest_ends, durations,
                                     demands, care_factors, no_precedences, predecessors,
                                     successors, prec_delays, max_demand,
                                     model_file, solver_choice, model_type, solver_type,
                                     run_costs[:], search, care_f_weight)
    try:
        sol = solver_results._solutions[-1].assignments
        obj = solver_results._solutions[-1].objective
        run_time = solver_results._solutions[-1].statistics['time'].microseconds / 1000
    except:
        print("error")
    optimised_d_profile, actual_starts_op = optimised_profile(sol, solver_type, demands, durations)


    return sactual_starts_ogsa, optimistic_d_profile, actual_starts_op, optimised_d_profile


def pricing_master_problem():

    return True


def iteration():

    # initialisation
    households = area_generation()

    return True

iteration()