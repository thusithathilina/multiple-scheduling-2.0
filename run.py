from scripts.experiment import *

repeat_num = 1
household_nums = [10]
new_data = ReadMode.OLD
type_cost_function = "piece-wise"
# type_cost_function = "linear"
algorithms_labels = dict()

algorithms_labels[k1_optimal] = dict()
algorithms_labels[k1_heuristic] = dict()
algorithms_labels[k1_optimal][k2_scheduling] = k1_optimal
algorithms_labels[k1_heuristic][k2_scheduling] = k1_heuristic
algorithms_labels[k1_optimal][k2_pricing] = "{}_fw".format(k1_optimal)
algorithms_labels[k1_heuristic][k2_pricing] = "{}_fw".format(k1_heuristic)

for n in household_nums:
    for _ in range(repeat_num):
        experiment(n, no_tasks, new_data, type_cost_function, algorithms_labels)
