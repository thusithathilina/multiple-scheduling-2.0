from bisect import bisect_left
from multiple.scripts.cfunctions import find_ge, find_le
from multiple.scripts.input_parameter import *
from math import ceil


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
        cost = round(cost, 2)

    return price_day, cost


def pricing_step_size(pricing_table, demand_profile_pre, demand_profile_new, cost_type, prices_pre, cost_pre):
    demand_profile_updated = demand_profile_new[:]
    cost = cost_pre

    # todo - I need to add the penalty cost to the equation
    # todo - the formula for the changed of cost is wrong.

    # simply compute the total consumption cost and the price
    if demand_profile_pre is None or prices_pre is None:
        price_day, cost = pricing_cost(demand_profile_new, pricing_table, cost_type)
        best_step_size = 1

    # apply the FW algorithm
    else:
        change_of_gradient = 999
        price_day = prices_pre[:]
        print("before fw", price_day)
        # demand_profile_changed = [d_n - d_p for d_n, d_p in zip(demand_profile_new, demand_profile_pre)]
        demand_profile_updated_pre = demand_profile_pre[:]
        best_step_size = 0
        change_of_cost = -999
        cost_update_pre = cost_pre
        counter = 0
        while change_of_cost < 0 and best_step_size < 1:
            step_profile = []
            for dp, dn, d_levels_period in \
                    zip(demand_profile_updated_pre, demand_profile_new, pricing_table[k0_demand_table].values()):
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
            demand_profile_updated_temp = [d_p + (d_n - d_p) * temp_step_size for d_p, d_n in
                                      zip(demand_profile_updated_pre, demand_profile_new)]
            price_updated, cost_updated = pricing_cost(demand_profile_updated_temp, pricing_table, cost_type)
            change_of_cost = cost_updated - cost_update_pre
            # change_of_gradient = sum([d_c * (p_n - p_p) for d_c, p_n, p_p in
            #                          zip(demand_profile_changed, price_updated, prices_pre)])
            # print(price_updated)
            # print("change of cost", change_of_cost)
            # change_of_gradient = abs(change_of_gradient)

            demand_profile_updated_pre = demand_profile_updated_temp[:]
            # print(change_of_cost < 0 and temp_step_size < 1)

            if change_of_cost < 0 and best_step_size < 1:
                best_step_size += temp_step_size
                demand_profile_updated = demand_profile_updated_temp[:]
                # print("best step size", best_step_size)
                price_day = price_updated[:]
                print("after fw", price_day)
                cost = cost_updated
                # print("cost", cost)
                counter += 1

        print("best step size found in", counter, "iterations")

    return demand_profile_updated, best_step_size, price_day, cost


def pricing_master_problem(iteration, pricing_table, area, cost_type):
    # read the newly optimised demand profile
    heuristic_demand_profile_new = area[k0_profile][k1_heuristic][iteration]  # newly optimised demand profile
    optimal_demand_profile_new = area[k0_profile][k1_optimal][iteration]

    if iteration == 0:
        heuristic_demand_profile_pre = None
        optimal_demand_profile_pre = None
        heuristic_prices_pre = None
        optimal_prices_pre = None
        heuristic_cost_pre = None
        optimal_cost_pre = None
    else:
        heuristic_demand_profile_pre = area[k0_profile][k1_heuristic_fw][iteration - 1]
        optimal_demand_profile_pre = area[k0_profile][k1_optimal_fw][iteration - 1]
        heuristic_prices_pre = area[k0_prices][k1_heuristic_fw][iteration - 1]
        optimal_prices_pre = area[k0_prices][k1_optimal_fw][iteration - 1]
        heuristic_cost_pre = area[k0_cost][k1_heuristic_fw][iteration - 1]
        optimal_cost_pre = area[k0_cost][k1_optimal_fw][iteration - 1]

    heuristic_demand_profile_updated, heuristic_best_step_size, heuristic_price_day, heuristic_cost \
        = pricing_step_size(pricing_table, heuristic_demand_profile_pre, heuristic_demand_profile_new,
                            cost_type, heuristic_prices_pre, heuristic_cost_pre)

    optimal_demand_profile_updated, optimal_best_step_size, optimal_price_day, optimal_cost \
        = pricing_step_size(pricing_table, optimal_demand_profile_pre, optimal_demand_profile_new,
                            cost_type, optimal_prices_pre, optimal_cost_pre)

    return heuristic_demand_profile_updated, heuristic_best_step_size, heuristic_price_day, heuristic_cost, \
        optimal_demand_profile_updated, optimal_best_step_size, optimal_price_day, optimal_cost
