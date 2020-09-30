from experiment import *


household_nums = [10]
new_data = True
type_cost_function = "piece-wise"
for n in household_nums:
    for _ in range(1):
        experiment(n, no_tasks, new_data, type_cost_function)
