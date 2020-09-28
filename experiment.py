from scripts.iteration import *
from scripts.input_parameter import *

household_nums = [500, 1000]
new_data = True
for hn in household_nums:
    iteration(hn, no_tasks, new_data)
