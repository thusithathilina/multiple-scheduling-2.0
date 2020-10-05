from scripts.data_generation import *
from scripts.household_scheduling import *
from scripts.drsp_pricing import *
from scripts.input_parameter import *
from scripts.cfunctions import *


def update_data(r_dict, itr, k1_alg, d, k0):
    if k0 not in r_dict[k1_alg]:
        r_dict[k1_alg][k0] = dict()
    if d is not None:
        r_dict[k1_alg][k0][itr] = d
    return r_dict


def update_area_data(area_dict, i, k1_algorithm, demands, prices, obj, cost, penalty, step_size, time):

    if demands is not None:
        max_demand = max(demands)
        area_dict[k1_algorithm][k0_demand][i] = demands
        area_dict[k1_algorithm][k0_demand_max][i] = max_demand
        area_dict[k1_algorithm][k0_demand_total][i] = sum(demands)
        area_dict[k1_algorithm][k0_par][i] = round(max_demand / average(demands), 2)

    area_dict = update_data(area_dict, i, k1_algorithm, prices, k0_prices)
    area_dict = update_data(area_dict, i, k1_algorithm, obj, k0_obj)
    area_dict = update_data(area_dict, i, k1_algorithm, cost, k0_cost)
    area_dict = update_data(area_dict, i, k1_algorithm, penalty, k0_penalty)
    area_dict = update_data(area_dict, i, k1_algorithm, step_size, k0_step)
    area_dict = update_data(area_dict, i, k1_algorithm, time, k0_time)

    return area_dict


def iteration(area, households, pricing_table, cost_type, str_summary, solvers, models, algorithm_label):

    key_scheduling = algorithm_label[k2_scheduling]
    key_pricing_fw = algorithm_label[k2_pricing]

    # 1.1 - the prices of the total preferred demand profile
    prices, cost, demands_fw, prices_fw, cost_fw, penalty_fw, step_fw, time_fw \
        = pricing_master_problem(0, pricing_table, area, cost_type, key_scheduling, key_pricing_fw)
    print("First day prices calculated...")

    # 1.2 - initialise tracker values: objective values, costs, penalties, steps and the run times
    area = update_area_data(area, 0, key_scheduling, None, prices,
                            cost, cost, None, step_fw, 0)
    area = update_area_data(area, 0, key_pricing_fw, demands_fw, prices_fw,
                            cost_fw, cost_fw, penalty_fw, step_fw, 0)
    print("The demand profile, prices, the cost, the step, and the run time initialised...")

    # 2 - iterations
    itr = 1
    solver_choice = solvers[solver_type]
    model_file = models[solver_type][model_type]
    step_fw = 1
    while step_fw > 0:

        time_scheduling_iteration = 0
        time_pricing_iteration = 0

        # ------------------------------ 1. rescheduling step ------------------------------ #
        demands_area_scheduling = [0] * no_intervals
        obj_area = 0
        penalty_area = 0

        # 2.1 - reschedule given the prices at iteration k - 1
        for key, household in households.items():
            # 2.1.1 - reschedule a household
            prices_fw_pre = area[key_pricing_fw][k0_prices][itr - 1]
            starts_household, demands_household, obj_household, penalty_household, time_household \
                = household_scheduling_subproblem(no_intervals, no_periods, no_intervals_periods,
                                                  household, care_f_weight, care_f_max, prices_fw_pre,
                                                  model_file, model_type,
                                                  solver_type, solver_choice, var_selection, val_choice,
                                                  key_scheduling)

            # 2.1.2 - update the area data trackers: demand profile, total objective value, total penalty and run time
            households[key][k0_starts][key_scheduling][itr] = starts_household
            demands_area_scheduling = [x + y for x, y in zip(demands_household, demands_area_scheduling)]
            obj_area += obj_household
            penalty_area += penalty_household
            time_scheduling_iteration += time_household
            print("household {0} at iteration {1}".format(key, itr))

        # 2.2 - save the rescheduled results
        demands = [sum(x) for x in grouper(demands_area_scheduling, no_intervals_periods)]
        area = update_area_data(area, itr, key_scheduling, demands, None,
                                obj_area, obj_area - penalty_area, penalty_area,
                                None, time_scheduling_iteration)

        # ------------------------------ 2. pricing step ------------------------------ #
        # 2.1 - apply the FW algorithm to calculate the prices, the step size and the cost
        prices, cost, demands_fw, prices_fw, cost_fw, penalty_fw, step_fw, time_fw \
            = pricing_master_problem(itr, pricing_table, area, cost_type, key_scheduling, key_pricing_fw)
        time_pricing_iteration += time_fw
        print("step size at iteration {}  = {}".format(itr, step_fw))

        # 2.3 - save pricing results
        area = update_area_data(area, itr, key_scheduling,
                                None, prices,
                                None, None, None, step_fw, None)
        area = update_area_data(area, itr, key_pricing_fw,
                                demands_fw, prices_fw,
                                cost_fw + penalty_fw, cost_fw, penalty_fw,
                                step_fw, time_pricing_iteration)

        # 3 - next iteration
        itr += 1

    return area, str_summary, itr - 1
