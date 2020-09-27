from multiple.scripts.data_generation import *
from multiple.scripts.household_scheduling import *
from multiple.scripts.drsp_pricing import *
from multiple.scripts.input_parameter import *
from multiple.scripts.output_results import write_results


def iteration():
    def average(lst):
        return sum(lst) / len(lst)

    def update_area_data(i, result_type, demand_profile, obj, cost, penalty, step_size, prices):
        area[k0_obj][result_type][i] = obj
        area[k0_cost][result_type][i] = cost
        area[k0_penalty][result_type][i] = penalty
        area[k0_ss][result_type][i] = step_size
        area[k0_prices][result_type][i] = prices
        area[k0_profile][result_type][i] = demand_profile
        max_demand = max(demand_profile)
        area[k0_demand_max][result_type][i] = max_demand
        area[k0_par][result_type][i] = round(average(demand_profile) / max_demand, 2)

    # 0 - initialise experiment (iteration = 0)
    print("---------- Experiment Summary ----------")
    str_summary = "{0} households, {1} tasks per household".format(no_households, no_tasks)
    print(str_summary)
    print("---------- Experiments begin! ----------")

    # 0.1 - generation household data and the total preferred demand profile
    if new_households:
        households, area = area_generation(no_intervals, no_periods, no_intervals_periods, file_household_area_folder,
                                           no_households, no_tasks, care_f_weight, care_f_max, max_demand_multiplier,
                                           file_probability, file_demand_list)
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
    print("Pricing table created...")

    # 0.3 - the prices of the total preferred demand profile
    heuristic_demands, heuristic_step, heuristic_prices, heuristic_cost, \
        optimal_demands, optimal_step, optimal_prices, optimal_cost \
        = pricing_master_problem(0, pricing_table, area, cost_type)
    print("First day prices calculated...")

    # 0.4 - initialise tracker values
    update_area_data(0, k1_heuristic_fw, heuristic_demands, heuristic_cost, heuristic_cost, 0, heuristic_step,
                     heuristic_prices)
    update_area_data(0, k1_optimal_fw, optimal_demands, optimal_cost, optimal_cost, 0, optimal_step, optimal_prices)

    print("---------- Initialisation done! ----------")

    # iteration = k where k > 0
    itr = 1
    total_time_heuristic = 0
    total_time_optimal = 0
    solver_choice = solvers[solver_type]
    model_file = models[solver_type][model_type]
    while optimal_step > 0 or heuristic_step > 0:

        # ---------- 1. rescheduling step ---------- #
        # 1.1 - reschedule given the prices at iteration k - 1
        heuristic_demands_scheduling = [0] * no_intervals
        optimal_demands_scheduling = [0] * no_intervals
        heuristic_obj_area = 0
        optimal_obj_area = 0
        for key, household in households.items():
            # 1.1.1 - reschedule a household
            heuristic_starts_household, heuristic_demands_household, heuristic_obj_household, heuristic_time_household, \
             optimal_starts_household, optimal_demands_household, optimal_obj_household, optimal_time_household \
                = household_scheduling_subproblem(no_intervals, no_tasks, no_periods, no_intervals_periods,
                                                  household, care_f_weight, care_f_max, area[k0_prices], itr,
                                                  model_file, model_type,
                                                  solver_type, solver_choice, var_selection, val_choice)

            # 1.1.2 - update the area demand profile
            heuristic_demands_scheduling \
                = [x + y for x, y in zip(heuristic_demands_household, heuristic_demands_scheduling)]
            optimal_demands_scheduling \
                = [x + y for x, y in zip(optimal_demands_household, optimal_demands_scheduling)]

            # 1.1.3 - update the total objective value
            heuristic_obj_area += heuristic_obj_household
            optimal_obj_area += optimal_obj_household

            # 1.1.4 - update the run time of rescheduling
            total_time_heuristic += heuristic_time_household
            total_time_optimal += optimal_time_household
            print("household {}".format(key))

        # 1.2 - process results
        # 1.2.1 - save the total rescheduling time of this iteration
        area[k0_time][k1_heuristic_fw] = total_time_heuristic
        area[k0_time][k1_optimal_fw] = total_time_optimal

        # 1.2.2 - convert the area demand profile for the pricing purpose
        heuristic_demands_pricing = [sum(x) for x in grouper(heuristic_demands_scheduling, no_intervals_periods)]
        optimal_demands_pricing = [sum(x) for x in grouper(optimal_demands_scheduling, no_intervals_periods)]

        # 1.2.3 - save the converted area demand profiles
        area[k0_profile][k1_heuristic][itr] = heuristic_demands_pricing
        area[k0_profile][k1_optimal][itr] = optimal_demands_pricing

        # 1.2.4 - save the total objective value


        # ---------- 2. pricing step ---------- #
        # 2.1 - calculate the prices, the step size and the cost after applying the FW algorithm
        heuristic_demands_fw, heuristic_step, heuristic_prices, heuristic_cost, \
            optimal_demand_fw, optimal_step, optimal_prices, optimal_cost \
            = pricing_master_problem(itr, pricing_table, area, cost_type)

        print("step size at iteration {}: heuristic = {}, optimal = {}"
              .format(itr, float(heuristic_step), float(optimal_step)))

        # 2.2 - save the demand profiles, prices and the step size at this iteration
        # todo - need to calculate the penalty
        update_area_data(itr, k1_heuristic_fw, heuristic_demands_fw, heuristic_cost, heuristic_cost, 0, heuristic_step,
                         heuristic_prices)
        update_area_data(itr, k1_optimal_fw, optimal_demand_fw, optimal_cost, optimal_cost, 0, optimal_step,
                         optimal_prices)

        # 3 - move on to the next iteration
        itr += 1

    print("---------- Result Summary ----------")
    print("Converged in {0} iteration".format(itr))
    print("---------- Iteration done! ----------")

    # 4 - process results
    output_date_time_folder = write_results(area, output_folder, str_summary)


iteration()
