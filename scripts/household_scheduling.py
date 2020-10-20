import timeit
from minizinc import *
import random as r
from scripts.input_parameter import *


def data_preprocessing(num_intervals, demands, prices_day, earliest_starts, latest_ends, durations,
                       preferred_starts, care_factors, cf_weight, cf_max):
    max_demand = max(demands)
    max_duration = max(durations)
    run_costs = []
    big_cost = (num_intervals * cf_max * max_demand + max_demand * max_duration)
    num_tasks = len(demands)

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
                try:
                    rc += sum([prices_day[j % num_intervals] for j in range(t, t + duration)]) * demand
                except IndexError:
                    print("Error: check the prices.")
            else:
                rc = big_cost
            run_cost_task.append(int(rc))
        run_costs.append(run_cost_task)
    return run_costs


def household_heuristic_solving(num_intervals, durations, demands, predecessors, successors,
                                succ_delays, max_demand, run_costs, preferred_starts, latest_ends, cf_max):
    start_time = timeit.default_timer()

    actual_starts = []
    household_profile = [0] * num_intervals
    obj = 0
    max_duration = max(durations)
    big_cost = (num_intervals * cf_max * max_demand + max_demand * max_duration)
    num_tasks = len(demands)

    def retrieve_successors_or_precedents(list0, prec_or_succ_list1, succ_prec_list2):
        list_r = []
        for l in list0:
            if l in prec_or_succ_list1:
                succ_or_prec_indices = [i2 for i2, k in enumerate(prec_or_succ_list1) if k == l]
                succ_or_prec = [succ_prec_list2[i2] for i2 in succ_or_prec_indices]
                succ_succ_or_prec_prec \
                    = retrieve_successors_or_precedents(succ_or_prec, prec_or_succ_list1, succ_prec_list2)
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
        earliest_suc_lstart_w_delay = 0
        earliest_suc_lstart = num_intervals - 1
        if bool(tasks_successors):
            suc_durations = [durations[i2] for i2 in tasks_successors]
            suc_lends = [latest_ends[i2] for i2 in tasks_successors]
            earliest_suc_lstart = min([lend - dur for lend, dur in zip(suc_lends, suc_durations)])
            # earliest_suc_lstart_w_delay = earliest_suc_lstart - succ_delay
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
            if task_costs[j] < big_cost and earliest_suc_lstart_w_delay < j < earliest_suc_lstart - duration + 1 \
                    and latest_pre_finish < j < latest_pre_finish_w_delay:
                feasible_intervals.append(j)

        try:
            feasible_min_cost = min([task_costs[f] for f in feasible_intervals])
            cheapest_intervals = [f for f in feasible_intervals if task_costs[f] == feasible_min_cost]
            a_start = cheapest_intervals[0]#r.choice(cheapest_intervals)

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
                # a_start = r.choice(feasible_min_cost_indices)
                a_start = feasible_min_cost_indices[0]

                temp_profile = household_profile[:]
                for d in range(a_start, a_start + duration):
                    temp_profile[d] += demand
                temp_max_demand = max(temp_profile)

            if len(feasible_intervals) == 0 and not max_demand_starts:
                a_start = min(max_demand_starts, key=max_demand_starts.get)

        except ValueError:
            # print("No feasible intervals left for task", task_id)
            a_start = preferred_starts[task_id]

        actual_starts.append(a_start)
        for d in range(a_start, a_start + duration):
            household_profile[d % num_intervals] += demand
        obj += run_costs[task_id][a_start]

    elapsed = timeit.default_timer() - start_time
    obj = round(obj, 2)

    return actual_starts, household_profile, obj, elapsed


def household_optimal_solving \
                (num_intervals, prices_day, preferred_starts, earliest_starts, latest_ends, durations,
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
    num_tasks = len(demands)
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

    return actual_starts, optimal_demand_profile, obj, time


def household_scheduling_subproblem \
                (num_intervals, num_periods, num_intervals_periods,
                 household, cf_weight, cf_max, prices,
                 model_file, m_type, s_type, solver_choice, var_sel, val_cho, k1_algorithm_scheduling):
    # extract household data
    key = household[k0_household_key]
    demands = household["demands"]
    durations = household["durs"]
    earliest_starts = household["ests"]
    latest_ends = household["lfts"]
    preferred_starts = household["psts"]
    care_factors = household["cfs"]
    precedents = [x[0] for x in list(household["precs"].values())]
    successors = list(household["precs"].keys())
    succ_delays = household["succ_delays"]  # need to change this format when sending it to the solver
    no_precedences = household["no_prec"]
    max_demand = household["demand"]["limit"]

    if len(prices) == num_periods:
        prices = [int(p) for p in prices[:num_periods] for _ in range(num_intervals_periods)]
    else:
        prices = [int(p) for p in prices]

    # data preprocessing
    run_costs = data_preprocessing(num_intervals, demands, prices,
                                   earliest_starts, latest_ends, durations, preferred_starts,
                                   care_factors, cf_weight, cf_max)

    if "heuristic" in k1_algorithm_scheduling:
        actual_starts, demands_new, obj, runtime \
            = household_heuristic_solving(num_intervals, durations, demands, precedents, successors,
                                          succ_delays, max_demand, run_costs, preferred_starts, latest_ends, cf_max)

    else:  # "optimal" in k1_algorithm
        succ_delays2 = [x[0] for x in list(household["succ_delays"].values())]
        search = "int_search(actual_starts, {}, {}, complete)".format(var_sel, val_cho)
        actual_starts, demands_new, obj, runtime \
            = household_optimal_solving(num_intervals, prices, preferred_starts, earliest_starts,
                                        latest_ends, durations, demands, care_factors, no_precedences, precedents,
                                        successors, succ_delays2, max_demand, model_file, solver_choice, m_type,
                                        s_type, run_costs, search, cf_weight)

    penalty = sum([abs(pst - ast) * cf_weight * cf for pst, ast, cf
                   in zip(preferred_starts, actual_starts, care_factors)])

    if key % 100 == 0:
        print("Household {} rescheduled by {}".format(key, k1_algorithm_scheduling))

    # household[k0_starts] = actual_starts (moved to outer iteration)

    return {k0_household_key: key, k0_starts: actual_starts, k0_demand: demands_new, k0_obj: obj, k0_penalty: penalty,
            k0_time: round(runtime, 3)}
