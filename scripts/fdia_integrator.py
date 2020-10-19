from more_itertools import grouper

from scripts.input_parameter import no_intervals, no_intervals_periods, k0_demand, k0_demand_total, k0_demand_max


# This method is use to inject fake demand from the 1st household
def fdia_inject(households, area, algorithm_label, injection_percentage=0):
    if injection_percentage == 0:
        return
    households[0]["o_demands"] = households[0]["demands"]
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
