# time related parameters
no_intervals = 144
no_periods = 48
no_intervals_periods = int(no_intervals / no_periods)

# household related parameters
new_households = True
new_households = False
no_households = 1000
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
file_pricing_table = 'inputs/pricing_table_0.csv'
file_household_area_folder = 'data/'

# demand profile related parameters
k0_profile = "profile"
k0_profile_updated = "profile_updated"
k1_interval = "interval"
k1_period = "period"
k1_optimal = "optimal"
k1_heuristic = "heuristic"


# step size
k0_ss = "step_size"

# objective related parameters
k0_cost = "cost"
k0_penalty = "inconvenient"
k0_obj = "obj"

# pricing related parameters
k0_prices = "price_history"
k0_price_levels = "price_levels"
k0_demand_table = "demand_levels"

