from bisect import bisect_left
from scripts.cfunctions import find_ge, find_le
from scripts.input_parameter import *
from time import time


def pricing_cost(demand_profile, pricing_table, cost_function):
    price_day = []
    cost = 0
    price_levels = pricing_table[k0_price_levels]

    for demand_period, demand_level_period in zip(demand_profile, pricing_table[k0_demand_table].values()):
        demand_level = list(demand_level_period.values())
        level = bisect_left(demand_level, demand_period)
        if level != len(demand_level):
            price = price_levels[level]
        else:
            price = price_levels[-1]
        price_day.append(price)

        if "piece-wise" in cost_function and level > 0:
            cost += demand_level[0] * price_levels[0]
            cost += (demand_period - demand_level[level - 1]) * price
            cost += sum([(demand_level[i] - demand_level[i - 1]) * price_levels[i] for i in range(1, level)])
        else:
            cost += demand_period * price
        cost = round(cost, 2)

    return price_day, cost


def pricing_step_size(pricing_table, demand_profile_pre, demand_profile_new, penalty_pre, penalty_new,
                      cost_type, prices_pre, cost_pre):
    cost = cost_pre
    penalty = penalty_pre
    change_of_penalty = penalty_new - penalty_pre
    price_day = prices_pre[:]

    # print("before fw", price_day)
    demand_profile_changed = [d_n - d_p for d_n, d_p in zip(demand_profile_new, demand_profile_pre)]
    demand_profile_fw = demand_profile_pre[:]
    demand_profile_fw_pre = demand_profile_pre[:]
    best_step_size = 0
    gradient = -999
    counter = 0
    while gradient < 0 and best_step_size < 1:
        step_profile = []
        for dp, dn, d_levels_period in \
                zip(demand_profile_fw_pre, demand_profile_new, pricing_table[k0_demand_table].values()):
            d_levels = list(d_levels_period.values())
            step = 1
            dd = dn - dp
            if dd != 0:
                try:
                    dl = find_ge(d_levels, dp) + 0.01 if dd > 0 else find_le(d_levels, dp)
                    step = round((dl - dp) / dd, 3)
                    step = min(1, step) if step > 0.01 else 1
                except ValueError:
                    pass
            step_profile.append(step)

        temp_step_size = min(step_profile)
        demand_profile_fw_temp = [d_p + (d_n - d_p) * temp_step_size for d_p, d_n in
                                  zip(demand_profile_fw_pre, demand_profile_new)]
        price_fw, cost_fw = pricing_cost(demand_profile_fw_temp, pricing_table, cost_type)

        gradient = sum([d_c * p_fw for d_c, p_fw in
                        zip(demand_profile_changed, price_fw)]) + change_of_penalty

        demand_profile_fw_pre = demand_profile_fw_temp[:]

        if gradient < 0 and best_step_size < 1:
            best_step_size += temp_step_size
            demand_profile_fw = demand_profile_fw_temp[:]
            # print("best step size", best_step_size)
            price_day = price_fw[:]
            # print("after fw", price_day)
            cost = cost_fw
            penalty = best_step_size * change_of_penalty
            # print("cost", cost)
            counter += 1

    print("best step size found in", counter, "iterations")

    return demand_profile_fw, best_step_size, price_day, cost, penalty


def save_results(results, k1_algorithm_scheduling, k1_algorithm_fw, prices, cost, demands_fw, prices_fw,
                 cost_fw, penalty_fw, step, run_t):
    results[k1_algorithm_scheduling] = dict()
    results[k1_algorithm_scheduling][k0_prices] = prices
    results[k1_algorithm_scheduling][k0_cost] = cost

    results[k1_algorithm_fw] = dict()
    results[k1_algorithm_fw][k0_demand] = demands_fw
    results[k1_algorithm_fw][k0_prices] = prices_fw
    results[k1_algorithm_fw][k0_cost] = cost_fw
    results[k1_algorithm_fw][k0_penalty] = penalty_fw
    results[k1_algorithm_fw][k0_step] = step
    results[k1_algorithm_fw][k0_time] = run_t

    return results


def pricing_master_problem(iteration, pricing_table, area, cost_function, alg_label):
    pricing_results = dict()
    key_scheduling = alg_label[k2_scheduling]
    key_pricing_fw = alg_label[k2_pricing]

    def procedure(k1_algorithm_scheduling, k1_algorithm_fw):
        # the new demand profile generated at the current iteration
        demands_new = area[k0_demand][k1_algorithm_scheduling][iteration]
        penalty_new = area[k0_penalty][k1_algorithm_scheduling][iteration]

        # calculate the prices and the cost of the new demand profile
        prices1, cost1 = pricing_cost(demands_new, pricing_table, cost_function)

        if iteration > 0:
            demands_fw_pre = area[k0_demand][k1_algorithm_fw][iteration - 1]
            prices_fw_pre = area[k0_prices][k1_algorithm_fw][iteration - 1]
            cost_fw_pre = area[k0_cost][k1_algorithm_fw][iteration - 1]
            penalty_fw_pre = area[k0_penalty][k1_algorithm_fw][iteration - 1]

            demands_fw1, step1, prices_fw1, cost_fw1, penalty_fw1 \
                = pricing_step_size(pricing_table, demands_fw_pre, demands_new,
                                    penalty_fw_pre, penalty_new,
                                    cost_function_type, prices_fw_pre, cost_fw_pre)
        else:
            demands_fw1 = demands_new
            prices_fw1 = prices1
            cost_fw1 = cost1
            penalty_fw1 = penalty_new
            step1 = 1

        return prices1, cost1, demands_fw1, prices_fw1, cost_fw1, penalty_fw1, step1

    # todo - the price to be hacked
    t_begin = time()
    prices, cost, demands_fw, prices_fw, cost_fw, penalty_fw, step = procedure(key_scheduling, key_pricing_fw)
    run_time = time() - t_begin
    pricing_results = save_results(pricing_results, key_scheduling, key_pricing_fw,
                                   prices, cost, demands_fw, prices_fw, cost_fw, penalty_fw, step, run_time)

    # optimal_prices, optimal_cost, optimal_demands_fw, optimal_prices_fw, \
    # optimal_cost_fw, optimal_penalty_fw, optimal_step = procedure(k1_optimal_scheduling, k1_optimal_fw)
    # pricing_results = save_results(pricing_results, k1_optimal_scheduling, k1_optimal_fw,
    #                                optimal_prices, optimal_cost,
    #                                optimal_demands_fw, optimal_prices_fw, optimal_cost_fw, optimal_penalty_fw,
    #                                optimal_step)

    return pricing_results

