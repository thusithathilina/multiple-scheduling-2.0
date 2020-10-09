# in household_scheduling.py
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

# in data_generation.py
# area[k0_profile][k1_interval] = dict()
# area[k0_profile][k1_period] = dict()
# area[k0_profile][k1_interval][0] = area_demand_profile
# area[k0_profile][k1_period][0] = [sum(x) for x in grouper(area_demand_profile, num_intervals_periods)]
# area[k0_profile][k1_optimal_updated][0] = area_demand_profile
# area[k0_profile][k1_heuristic_updated][0] = area_demand_profile

# change_of_cost = cost_updated - cost_update_pre
# change_of_gradient = sum([d_c * (p_n - p_p) for d_c, p_n, p_p in
#                          zip(demand_profile_changed, price_updated, prices_pre)])
# print(price_updated)
# print("change of cost", change_of_cost)
# change_of_gradient = abs(change_of_gradient)

# in data_generation.py

# def initialise_area_trackers(k0_key, k1_key):
#     if k0_key not in area:
#         area[k0_key] = dict()
#     area[k0_key][k1_key] = dict()
#
#
# # initialise trackers
# area_demand_profile_pricing = [sum(x) for x in grouper(area_demand_profile, num_intervals_periods)]
# # track four types of demand profiles, prices, objective values, costs, penalties, max demands and PARs
# k0_keys = [k0_demand, k0_prices, k0_obj, k0_cost, k0_penalty, k0_time, k0_step]
# # k1_keys = [k1_optimal_scheduling, k1_heuristic_scheduling, k1_optimal_fw, k1_heuristic_fw]
# area[k0_demand_max] = dict()
# area[k0_demand_total] = dict()
# area[k0_par] = dict()
# for k0 in k0_keys:
#     for alg in algorithms_labels.values():
#         for k1 in alg.values():
#             initialise_area_trackers(k0, k1)
#             # initial values for four kinds of demand profiles, max demands, PARs and the penalty
#             if k0 == k0_demand:
#                 area[k0][k1][0] = area_demand_profile_pricing
#                 max_demand = max(area_demand_profile_pricing)
#                 area[k0_demand_max][k1] = dict()
#                 area[k0_demand_total][k1] = dict()
#                 area[k0_par][k1] = dict()
#                 area[k0_demand_max][k1][0] = max_demand
#                 area[k0_demand_total][k1][0] = sum(area_demand_profile_pricing)
#                 area[k0_par][k1][0] = average(area_demand_profile_pricing) / max_demand
#
#             if k0 == k0_penalty:
#                 area[k0_penalty][k1][0] = 0
#
#             if k0 == k0_step:
#                 if "fw" in k1:
#                     initialise_area_trackers(k0, k1)

# in household_scheduling.py
# def save_results(results, k1_algorithm_scheduling, return_dict):
#     results[k1_algorithm_scheduling] = dict()
#     results[k1_algorithm_scheduling][k0_starts] = return_dict[k0_starts]
#     results[k1_algorithm_scheduling][k0_demand] = return_dict[k0_demand]
#     results[k1_algorithm_scheduling][k0_obj] = return_dict[k0_obj]
#     results[k1_algorithm_scheduling][k0_penalty] = return_dict[k0_penalty]
#     results[k1_algorithm_scheduling][k0_time] = return_dict[k0_time]
#
#     return results

# in iteration.py
# def extract_rescheduling_results(k1_algorithm_scheduling, results):
#     starts_t = results[k1_algorithm_scheduling][k0_starts]
#     demands_new_t = results[k1_algorithm_scheduling][k0_demand]
#     obj_t = results[k1_algorithm_scheduling][k0_obj]
#     penalty_t = results[k1_algorithm_scheduling][k0_penalty]
#     time_t = results[k1_algorithm_scheduling][k0_time]
#     return starts_t, demands_new_t, obj_t, penalty_t, time_t

# def extract_pricing_results(k1_algorithm_scheduling, k1_algorithm_fw, results):
#     prices_t = results[k1_algorithm_scheduling][k0_prices]
#     cost_t = results[k1_algorithm_scheduling][k0_cost]
#
#     demands_fw_t = results[k1_algorithm_fw][k0_demand]
#     prices_fw_t = results[k1_algorithm_fw][k0_prices]
#     cost_fw_t = results[k1_algorithm_fw][k0_cost]
#     penalty_fw_t = results[k1_algorithm_fw][k0_penalty]
#     step_fw_t = results[k1_algorithm_fw][k0_step]
#     time_fw_t = results[k1_algorithm_fw][k0_time]
#
#     return prices_t, cost_t, demands_fw_t, prices_fw_t, cost_fw_t, penalty_fw_t, step_fw_t, time_fw_t

# in drsp_pricing.py
# def save_results(results, k1_algorithm_scheduling, k1_algorithm_fw, prices, cost, demands_fw, prices_fw,
#                  cost_fw, penalty_fw, step, run_t):
#     results[k1_algorithm_scheduling] = dict()
#     results[k1_algorithm_scheduling][k0_prices] = prices
#     results[k1_algorithm_scheduling][k0_cost] = cost
#
#     results[k1_algorithm_fw] = dict()
#     results[k1_algorithm_fw][k0_demand] = demands_fw
#     results[k1_algorithm_fw][k0_prices] = prices_fw
#     results[k1_algorithm_fw][k0_cost] = cost_fw
#     results[k1_algorithm_fw][k0_penalty] = penalty_fw
#     results[k1_algorithm_fw][k0_step] = step
#     results[k1_algorithm_fw][k0_time] = run_t
#
#     return results

# in output_results.py
# def dict_to_pd_dt(k0_ks, k1_ks):
#     for k0 in k0_ks:
#         if k0 in area_res:
#             for k1 in k1_ks:
#                 if k1 in area_res[k0]:
#                     df = pd.DataFrame.from_dict(area_res[k0][k1], orient='index')
#                     df.to_csv(exp_folder + "{}-{}.csv".format(k0, k1))

# in iteration.py
# for key, household in households.items():
#     # 2.1.1 - reschedule a household
#     starts_household, demands_household, obj_household, penalty_household, time_household \
#         = household_scheduling_subproblem(no_intervals, no_periods, no_intervals_periods,
#                                           household, care_f_weight, care_f_max, prices_fw_pre,
#                                           model_file, model_type,
#                                           solver_type, solver_choice, var_selection, val_choice,
#                                           key_scheduling)
