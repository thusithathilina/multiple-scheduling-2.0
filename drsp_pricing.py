from bisect import bisect_left
from multiple.cfunctions import find_ge, find_le
from multiple.fixed_parameter import *


def pricing_cost(demand_profile, pricing_table, cost_function_type):
    price_day = []
    cost = 0
    price_levels = pricing_table[k0_price_levels]

    for d, demand_level in zip(demand_profile, pricing_table[k0_demand_table]):
        level = bisect_left(demand_level, d)
        if level != len(demand_level):
            price = price_levels[level]
        else:
            price = price_levels[-1]
        price_day.append(price)

        if "piece-wise" in cost_function_type and level > 0:
            cost += demand_level[0] * price_levels[0]
            cost += (d - demand_level[level - 1]) * price
            cost += sum([(demand_level[i] - demand_level[i - 1]) * price_levels[i] for i in range(1, level)])
        else:
            cost += d * price

    return price_day, cost


def pricing_step_size(pricing_table, demand_profile_pre, demand_profile_new, cost_type):
    best_step_size = 1
    demand_profile_updated = demand_profile_new

    # simply compute the total consumption cost and the price
    if demand_profile_pre is None:
        price_day, cost = pricing_cost(demand_profile_new, pricing_table, cost_type)

    # apply the FW algorithm
    else:
        step_profile = []
        for dp, dn, d_levels in zip(demand_profile_pre, demand_profile_new, pricing_table[k0_demand_table]):
            step = 1
            dd = dn - dp
            if dd != 0:
                dl = find_ge(d_levels, dp) if dd > 0 else find_le(d_levels, dp)
                step = (dl - dp) / dd
            step_profile.append(step)

        best_step_size = min(step_profile)
        demand_profile_updated = [dp + (dn - dp) * best_step_size for dp, dn in
                                  zip(demand_profile_pre, demand_profile_new)]
        price_day, cost = pricing_cost(demand_profile_updated, pricing_table, cost_type)

    return demand_profile_updated, best_step_size, price_day, cost


def pricing_master_problem(iteration, pricing_table, area, cost_type):
    heuristic_demand_profile_new = area[k0_profile][k1_heuristic][iteration]  # newly optimised demand profile
    optimal_demand_profile_new = area[k0_profile][k1_optimal][iteration]

    if iteration == 0:
        heuristic_demand_profile_pre = None
        optimal_demand_profile_pre = None
    else:
        heuristic_demand_profile_pre = area[k0_profile][k1_heuristic][iteration - 1]
        optimal_demand_profile_pre = area[k0_profile][k1_optimal][iteration - 1]

    heuristic_demand_profile_updated, heuristic_best_step_size, heuristic_price_day, heuristic_cost \
        = pricing_step_size(pricing_table, heuristic_demand_profile_pre, heuristic_demand_profile_new, cost_type)

    optimal_demand_profile_updated, optimal_best_step_size, optimal_price_day, optimal_cost \
        = pricing_step_size(pricing_table, optimal_demand_profile_pre, optimal_demand_profile_new, cost_type)

    return heuristic_demand_profile_updated, heuristic_best_step_size, heuristic_price_day, heuristic_cost, \
        optimal_demand_profile_updated, optimal_best_step_size, optimal_price_day, optimal_cost
