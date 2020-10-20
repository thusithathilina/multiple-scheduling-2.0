from datetime import datetime
import os

from more_itertools import grouper

from scripts.drsp_pricing import pricing_cost
from scripts.input_parameter import no_intervals, no_intervals_periods, k0_demand, k0_demand_total, k0_demand_max, \
    k0_prices, k0_penalty, attack_result_folder, k0_cost


# This method is use to inject fake demand from the 1st household
def fdia_inject(households, area, algorithm_label, injection_percentage):
    households[0]["o_demands"] = households[0]["demands"][:]
    device_original_demand = households[0]["demands"][0]
    total_demand = area[algorithm_label][k0_demand_total][0]
    device_new_demand = device_original_demand + (total_demand * injection_percentage) / households[0]["durs"][0]
    households[0]["demands"][0] = device_new_demand

    # calculate the new demand profile of the attacker house
    household_profile = [0] * no_intervals
    device = 0
    for device_demand in households[0]['demands']:
        for device_duration in range(households[0]['durs'][device]):
            household_profile[(households[0]['psts'][device] + device_duration) % no_intervals] += device_demand
        device += 1

    # calculate the new community demand profile
    o_demand_profile = [sum(x) for x in grouper(households[0]['demand']['preferred'], no_intervals_periods)]
    n_demand_profile = [sum(x) for x in grouper(household_profile, no_intervals_periods)]

    area[algorithm_label][k0_demand][0] = [a - b for a, b in zip(area[algorithm_label][k0_demand][0], o_demand_profile)]
    area[algorithm_label][k0_demand][0] = [a + b for a, b in zip(area[algorithm_label][k0_demand][0], n_demand_profile)]
    area[algorithm_label + '_fw'][k0_demand][0] = area[algorithm_label][k0_demand][0]

    area[algorithm_label][k0_demand_total][0] = sum(area[algorithm_label][k0_demand][0])
    area[algorithm_label + '_fw'][k0_demand_total][0] = sum(area[algorithm_label][k0_demand][0])

    # area[algorithm_label][k0_demand_max][0] = max(area[algorithm_label][k0_demand][0])
    # area[algorithm_label + '_fw'][k0_demand_max][0] = max(area[algorithm_label][k0_demand][0])

    return households, area


def fdia_analyse_impact(households, area, algorithm_label, num_iterations, pricing_table, cost_function,
                        inject_percentage, attack_result_file_prepend):
    optimal_demand_with_attack = area[algorithm_label + '_fw'][k0_demand][num_iterations]
    optimal_price_with_attack = area[algorithm_label + '_fw'][k0_prices][num_iterations]

    if inject_percentage > 0:
        tmp_attack_device_profile_long = [0] * no_intervals
        tmp_real_device_profile_long = [0] * no_intervals
        for device_duration in range(households[0]['durs'][0]):
            tmp_attack_device_profile_long[(households[0]['psts'][0] + device_duration) % no_intervals] \
                += households[0]['demands'][0]
            tmp_real_device_profile_long[(households[0]['psts'][0] + device_duration) % no_intervals] \
                += households[0]['o_demands'][0]
        attacked_device_demand_profile = [sum(x) for x in grouper(tmp_attack_device_profile_long, no_intervals_periods)]
        real_device_demand_profile = [sum(x) for x in grouper(tmp_real_device_profile_long, no_intervals_periods)]

        # calculate the community demand profile without fake demand
        attack_free_demand = [a - b for a, b in zip(optimal_demand_with_attack, attacked_device_demand_profile)]
        attack_free_demand = [a + b for a, b in zip(attack_free_demand, real_device_demand_profile)]

        attack_free_price, attack_free_cost = pricing_cost(attack_free_demand, pricing_table, cost_function)
        attack_free_price_long = [p for p in attack_free_price for i in range(no_intervals_periods)]
    else:
        attack_free_demand = optimal_demand_with_attack
        attack_free_price = optimal_price_with_attack
        attack_free_price_long = [p for p in optimal_price_with_attack for i in range(no_intervals_periods)]
        attack_free_cost = area[algorithm_label + '_fw'][k0_cost][num_iterations]

    if inject_percentage > 0:
        load_per_schedule_period = (households[0]['o_demands'][0]) / (no_intervals / 24.0)
    else:
        load_per_schedule_period = (households[0]['demands'][0]) / (no_intervals / 24.0)
    start = households[0]['psts'][0]
    duration = households[0]['durs'][0]
    device_cost = (load_per_schedule_period * sum(attack_free_price_long[start:start + duration])) / 100

    demand_change = (area[algorithm_label][k0_demand_total][0] - sum(attack_free_demand))/area[algorithm_label][k0_demand_total][0]
    penalty = area[algorithm_label][k0_penalty][num_iterations]

    # Create attack result folder
    attack_result_base_folder = attack_result_folder + datetime.now().strftime("%y-%m-%d") + "/"
    if not os.path.exists(attack_result_base_folder):
        os.makedirs(attack_result_base_folder)

    file_path = attack_result_base_folder + attack_result_file_prepend + '_attack_impact.csv'
    with open(file_path, 'a') as f:
        f.write("Attack-" + str(len(households)) + "-" + str(start) + "-" + str(duration)
                + "," + str(inject_percentage)
                + "," + str(demand_change * 100) + "%"
                + "," + str(device_cost)
                + "," + str(attack_free_cost)
                + "," + str(penalty)
                + "," + str(attack_free_cost + penalty)
                + "\n")
        f.close()

    optimal_demand_with_attack = [x / 1000 for x in optimal_demand_with_attack]
    attack_free_demand = [x / 1000 for x in attack_free_demand]
    optimal_price_with_attack = [x / 100 for x in optimal_price_with_attack]
    attack_free_price = [x / 100 for x in attack_free_price]

    file_path = attack_result_base_folder + attack_result_file_prepend + '_demands.csv'
    with open(file_path, 'a') as f:
        f.write("Attack-" + str(len(households)) + "-" + str(start) + "-" + str(duration) + "-" + str(inject_percentage)
                + ",Optimal," + str(optimal_demand_with_attack)[1:-1].replace(" ", "")
                + "\n")
        f.write("Attack-" + str(len(households)) + "-" + str(start) + "-" + str(duration)
                + ",Real," + str(attack_free_demand)[1:-1].replace(" ", "")
                + "\n")
        f.close()

    file_path = attack_result_base_folder + attack_result_file_prepend + '_prices.csv'
    with open(file_path, 'a') as f:
        f.write("Attack-" + str(len(households)) + "-" + str(start) + "-" + str(duration) + "-" + str(inject_percentage)
                + ",Optimal," + str(optimal_price_with_attack)[1:-1].replace(" ", "")
                + "\n")
        f.write("Attack-" + str(len(households)) + "-" + str(start) + "-" + str(duration)
                + ",Real," + str(attack_free_price)[1:-1].replace(" ", "")
                + "\n")
        f.close()
    print()
