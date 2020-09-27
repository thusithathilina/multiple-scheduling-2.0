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
