# time related parameters
no_intervals = 144
no_periods = 48
no_intervals_periods = int(no_intervals / no_periods)

# household related parameters
new_households = True
new_households = False
no_households = 100
no_tasks = 5
max_demand_multiplier = no_tasks
care_f_max = 10
care_f_weight = 10

# pricing related parameters
pricing_table_weight = 1.0
cost_type = "linear"
# cost_type = "piece-wise"
zero_digit = 2

# solver related parameters
var_selection = "smallest"
val_choice = "indomain_min"
model_type = "pre"
solver_type = "cp"

# external file related parameters
file_cp_pre = 'models/Household-cp-pre.mzn'
file_cp_ini = 'models/Household-cp.mzn'
file_pricing_table = 'data/pricing_table_0.csv'
file_household_area_folder = 'data/'
file_probability = 'data/probability.csv'
file_demand_list = 'data/demands_list.csv'

# demand related parameters
# k0_starts = "start_times"
k0_demand = "demands"
k0_demand_max = "max_demand"
k0_demand_total = "total_demand"
k0_par = "PAR"
# step size
k0_step = "step_size"
# objective related parameters
k0_cost = "cost"
k0_penalty = "inconvenient"
k0_obj = "objective"
# pricing related parameters
k0_prices = "prices"
k0_price_levels = "price_levels"
k0_demand_table = "demand_levels"
# run time related
k0_time = "reschedule_time"

# k1_interval = "interval"
# k1_period = "period"
k1_optimal_scheduling = "optimal"
k1_heuristic_scheduling = "heuristic"
k1_optimal_fw = "optimal_fw"
k1_heuristic_fw = "heuristic_fw"

# result related parameters
output_folder = "results/"

