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


def read_data():
    prices = genfromtxt('inputs/prices.csv', delimiter=',', dtype="float").astype(int) # in cents
    model = 'models/Household-cp-pre.mzn'
    solver = "Gecode"

    return prices, model, solver


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

    print(" --- Household made ---")

    return preferred_starts, earliest_starts, latest_ends, durations, demands, care_factors, \
           no_precedences, predecessors, successors, prec_delays, maximum_demand, aggregated_loads


def area_generation():
    probability_demand_profile = genfromtxt('inputs/probability.csv', delimiter=',', dtype="float")


    households = dict()
    tasks = dict()

    for h in range(no_households):

        preferred_starts, earliest_starts, latest_ends, durations, demands, care_factors, \
        num_precedences, predecessors, successors, prec_delays, max_demand, household_profile \
            = household_generation(probability_demand_profile)

        household_key = h
        tasks[0, household_key, "demands"] = demands
        tasks[0, household_key, "durations"] = durations
        tasks[0, household_key, "earliest_starts"] = earliest_starts
        tasks[0, household_key, "latest_ends"] = latest_ends
        tasks[0, household_key, "care_factors"] = [cf * care_f_weight for cf in care_factors]
        tasks[0, household_key, "predecessors"] = predecessors + [-1] * (no_tasks - num_precedences)
        tasks[0, household_key, "successors"] = successors + [-1] * (no_tasks - num_precedences)
        tasks[0, household_key, "prec_delays"] = prec_delays + [-1] * (no_tasks - num_precedences)
        tasks[0, household_key, "preferred_starts"] = preferred_starts

        households[household_key] = dict()
        households[household_key]["profile", "preferred"] = household_profile
        households[household_key]["max", "preferred"] = max(household_profile)
        households[household_key]["max","limit"] = max_demand


def household_scheduling_subproblem(prices, model, solver):


