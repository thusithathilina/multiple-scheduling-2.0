from _csv import reader

from more_itertools import grouper
from scripts.cfunctions import average

from scripts.input_parameter import no_intervals, k0_household_key, k0_starts, k0_cost, \
    k0_penalty, k0_obj, k0_demand, k0_demand_max, k0_demand_total, k0_par


def convert(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


# This method can be used to convert the previous jobs.csv file to new structure
def convert_from_csv(path, algorithms_labels, num_intervals_periods, care_f_max=10, delete_house_no=True, kw_to_watt=True):
    with open(path, mode='r') as file:
        csv_reader = reader(file)
        headers = [h.strip(" \'") for h in next(csv_reader)]
        if delete_house_no:
            del headers[0]

        demand_multiplier = 1
        if kw_to_watt:
            demand_multiplier = 1000

        households = dict()
        community = []
        household = []
        for row in csv_reader:
            if len(row) == 0:
                continue
            if int(row[1]) == 0 and int(row[0]) > 0 and len(household) > 0:
                community.append(household)
                household = []
            if delete_house_no:
                del row[0]
            task = {h: convert(i) for h, i in zip(headers[:len(row)], row)}
            household.append(task)
        community.append(household)
        file.close()

    household_profile = [0] * no_intervals
    area_demand_profile = [0] * no_intervals
    h = 0
    for household in community:
        household_key = h
        households[household_key] = dict()

        households[household_key][k0_household_key] = household_key
        households[household_key]["demands"] = [int(job['demand'] * demand_multiplier) for job in household]
        households[household_key]["durs"] = [job['dur'] for job in household]
        households[household_key]["ests"] = [job['estart'] for job in household]
        households[household_key]["lfts"] = [job['lfinish'] for job in household]
        households[household_key]["psts"] = [job['pstart'] for job in household]
        households[household_key]["cfs"] = [job['caf'] * care_f_max for job in household]
        households[household_key]["precs"] = dict()
        households[household_key]["succ_delays"] = dict
        households[household_key]["no_prec"] = 0

        for job in household:
            for d in range(job['dur']):
                household_profile[(job['pstart'] + d) % no_intervals] += int(job['demand'] * demand_multiplier)

        households[household_key]["demand"] = dict()
        households[household_key]["demand"]["preferred"] = household_profile
        households[household_key]["demand"]["max"] = max(household_profile)
        households[household_key]["demand"]["limit"] = max(household_profile)

        households[household_key][k0_starts] = dict()
        households[household_key][k0_cost] = dict()
        households[household_key][k0_penalty] = dict()
        households[household_key][k0_obj] = dict()

        for k in algorithms_labels.keys():
            households[household_key][k0_starts][k] = dict()
            households[household_key][k0_starts][k][0] = households[household_key]["psts"]

        area_demand_profile = [x + y for x, y in zip(household_profile, area_demand_profile)]
        household_profile = [0] * no_intervals
        h += 1

    area_demand_profile2 = [sum(x) for x in grouper(area_demand_profile, num_intervals_periods)]
    max_demand = max(area_demand_profile2)
    total_demand = sum(area_demand_profile2)
    par = round(max_demand / average(area_demand_profile2), 2)
    area = dict()
    for k1, v1 in algorithms_labels.items():
        for v2 in v1.values():
            area[v2] = dict()
            area[v2][k0_demand] = dict()
            area[v2][k0_demand_max] = dict()
            area[v2][k0_demand_total] = dict()
            area[v2][k0_par] = dict()
            area[v2][k0_penalty] = dict()

            area[v2][k0_demand][0] = area_demand_profile2
            area[v2][k0_demand_max][0] = max_demand
            area[v2][k0_demand_total][0] = total_demand
            area[v2][k0_par][0] = par
            area[v2][k0_penalty][0] = 0

    return households, area
