from bisect import bisect_left
from multiple.cfunctions import find_ge, find_le
from multiple.fixed_parameter import *


def pricing_cost(demand_profile, pricing_table, cost_function_type):
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

        if "piece-wise" in cost_function_type and level > 0:
            cost += demand_level[0] * price_levels[0]
            cost += (demand_period - demand_level[level - 1]) * price
            cost += sum([(demand_level[i] - demand_level[i - 1]) * price_levels[i] for i in range(1, level)])
        else:
            cost += demand_period * price

    return price_day, cost


def pricing_step_size(pricing_table, demand_profile_pre, demand_profile_new, cost_type, prices_pre):
    best_step_size = 0
    demand_profile_updated = demand_profile_new
    cost = 0

    # simply compute the total consumption cost and the price
    if demand_profile_pre is None or prices_pre is None:
        price_day, cost = pricing_cost(demand_profile_new, pricing_table, cost_type)
        best_step_size = 1

    # apply the FW algorithm
    else:
        change_of_gradient = 999
        price_day = prices_pre
        step_profile = []
        while change_of_gradient > 1 and best_step_size < 1:
            for dp, dn, d_levels_period in \
                    zip(demand_profile_pre, demand_profile_new, pricing_table[k0_demand_table].values()):
                d_levels = list(d_levels_period.values())
                step = 1
                dd = dn - dp
                if dd != 0:
                    try:
                        dl = find_ge(d_levels, dp) if dd > 0 else find_le(d_levels, dp)
                        step = (dl - dp) / dd
                        step = min(1, step)
                    except ValueError:
                        pass  # keep step = 1
                step_profile.append(step)

            best_step_size = min(step_profile)
            demand_profile_updated = [dp + (dn - dp) * best_step_size for dp, dn in
                                      zip(demand_profile_pre, demand_profile_new)]
            price_day, cost = pricing_cost(demand_profile_updated, pricing_table, cost_type)
            change_of_gradient = sum([(d_n - d_p) * (p_n - p_p) for d_n, d_p, p_n, p_p in
                                      zip(demand_profile_new, demand_profile_pre, price_day, prices_pre)])
            prices_pre = price_day

    return demand_profile_updated, best_step_size, price_day, cost


def pricing_master_problem(iteration, pricing_table, area, cost_type):
    heuristic_demand_profile_new = area[k0_profile][k1_heuristic][iteration]  # newly optimised demand profile
    optimal_demand_profile_new = area[k0_profile][k1_optimal][iteration]

    if iteration == 0:
        heuristic_demand_profile_pre = None
        optimal_demand_profile_pre = None
        heuristic_prices_pre = None
        optimal_prices_pre = None
    else:
        heuristic_demand_profile_pre = area[k0_profile][k1_heuristic][iteration - 1]
        optimal_demand_profile_pre = area[k0_profile][k1_optimal][iteration - 1]
        heuristic_prices_pre = area[k0_prices][k1_heuristic][iteration - 1]
        optimal_prices_pre = area[k0_prices][k1_optimal][iteration - 1]

    heuristic_demand_profile_updated, heuristic_best_step_size, heuristic_price_day, heuristic_cost \
        = pricing_step_size(pricing_table, heuristic_demand_profile_pre, heuristic_demand_profile_new,
                            cost_type, heuristic_prices_pre)

    optimal_demand_profile_updated, optimal_best_step_size, optimal_price_day, optimal_cost \
        = pricing_step_size(pricing_table, optimal_demand_profile_pre, optimal_demand_profile_new,
                            cost_type, heuristic_prices_pre)

    return heuristic_demand_profile_updated, heuristic_best_step_size, heuristic_price_day, heuristic_cost, \
        optimal_demand_profile_updated, optimal_best_step_size, optimal_price_day, optimal_cost
