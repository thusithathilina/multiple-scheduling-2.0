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


def iteration(num_tasks_min, area, households, pricing_table, cost_type, str_summary, solvers, models, algorithm_label):

    def extract_pricing_results(k1_algorithm_scheduling, k1_algorithm_fw, results):
        prices_t = results[k1_algorithm_scheduling][k0_prices]
        cost_t = results[k1_algorithm_scheduling][k0_cost]

        demands_fw_t = results[k1_algorithm_fw][k0_demand]
        prices_fw_t = results[k1_algorithm_fw][k0_prices]
        cost_fw_t = results[k1_algorithm_fw][k0_cost]
        penalty_fw_t = results[k1_algorithm_fw][k0_penalty]
        step_fw_t = results[k1_algorithm_fw][k0_step]
        return prices_t, cost_t, demands_fw_t, prices_fw_t, cost_fw_t, penalty_fw_t, step_fw_t

    def extract_rescheduling_results(k1_algorithm_scheduling, results):
        starts_t = results[k1_algorithm_scheduling][k0_starts]
        demands_new_t = results[k1_algorithm_scheduling][k0_demand]
        obj_t = results[k1_algorithm_scheduling][k0_obj]
        penalty_t = results[k1_algorithm_scheduling][k0_penalty]
        time_t = results[k1_algorithm_scheduling][k0_time]
        return starts_t, demands_new_t, obj_t, penalty_t, time_t

    # 0.1 - the prices of the total preferred demand profile
    pricing_results = pricing_master_problem(0, pricing_table, area, cost_type, algorithm_label)
    print("First day prices calculated...")

    # 0.4 - initialise tracker values: objective values, costs, penalties, steps and the run times
    key_scheduling = algorithm_label[k2_scheduling]
    key_pricing_fw = algorithm_label[k2_pricing]
    prices, cost, demands_fw, prices_fw, cost_fw, penalty_fw, step_fw \
        = extract_pricing_results(key_scheduling, key_pricing_fw, pricing_results)
    area = update_area_data(area, 0, key_scheduling, None, prices,
                            cost, cost, None, None, 0)
    area = update_area_data(area, 0, key_pricing_fw, demands_fw, prices_fw,
                            cost_fw, cost_fw, penalty_fw, step_fw, 0)

    print("The demand profile, prices, the cost, the step, and the run time initialised...")

    print("---------- Initialisation done! ----------")

    # iteration = k where k > 0
    itr = 1
    solver_choice = solvers[solver_type]
    model_file = models[solver_type][model_type]

    total_time_scheduling = 0
    total_time_pricing = 0
    step_fw = 1

    while step_fw > 0:

        # -------------------- 1. rescheduling step -------------------- #
        demands_area_scheduling = [0] * no_intervals
        obj_area = 0
        penalty_area = 0

        # 1.1 - reschedule given the prices at iteration k - 1
        for key, household in households.items():

            # 1.1.1 - reschedule a household
            rescheduling_results \
                = household_scheduling_subproblem(no_intervals, no_periods, no_intervals_periods,
                                                  household, care_f_weight, care_f_max, area[k0_prices], itr,
                                                  model_file, model_type,
                                                  solver_type, solver_choice, var_selection, val_choice,
                                                  algorithm_label)

            # 1.1.2 - heuristic: extract results and update the demand profiles, total objective value and the runtime
            starts, demands_household, obj_household, penalty_household, time_household \
                = extract_rescheduling_results(key_scheduling, rescheduling_results)
            households[key][k0_starts][key_scheduling][itr] = starts
            demands_area_scheduling \
                = [x + y for x, y in zip(demands_household, demands_area_scheduling)]
            obj_area += obj_household
            penalty_area += penalty_household
            total_time_scheduling += time_household

            print("household {}".format(key))

        # 1.2 - process and save results
        # 1.2.1 - convert the area demand profile for the pricing purpose
        # and save the converted area demand profiles, total objective value and the total rescheduling time
        demands = [sum(x) for x in grouper(demands_area_scheduling, no_intervals_periods)]
        area = update_area_data(area, itr, key_scheduling, demands, None,
                                obj_area, obj_area - penalty_area, penalty_area,
                                None, total_time_scheduling)

        # -------------------- 2. pricing step -------------------- #
        # 2.1 - calculate the prices, the step size and the cost after applying the FW algorithm
        pricing_results_fw = pricing_master_problem(itr, pricing_table, area, cost_type, algorithm_label)

        # 2.2 - save results
        # 2.2.1 - save the demand profiles, prices and the step size at this iteration
        prices, cost, demands_fw, prices_fw, cost_fw, penalty_fw, step_fw \
            = extract_pricing_results(key_scheduling, key_pricing_fw, pricing_results_fw)
        area = update_area_data(area, itr, key_scheduling,
                                None, prices,
                                None, None, None, None, total_time_scheduling)
        area = update_area_data(area, itr, key_pricing_fw,
                                demands_fw, prices_fw,
                                cost_fw + penalty_fw, cost_fw, penalty_fw,
                                step_fw, total_time_scheduling)

        print("step size at iteration {}  = {}" .format(itr, step_fw))

        # 3 - move on to the next iteration
        itr += 1

    print("---------- Result Summary ----------")
    print("Converged in {0} iteration".format(itr - 1))
    print("---------- Iteration done! ----------")

    return area, str_summary

