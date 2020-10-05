from scripts.experiment import *
from datetime import date, datetime
import pandas as pd


repeat_num = 2
household_nums = [10, 50, 100]
new_data = True
# new_data = False
type_cost_function = "piece-wise"
# type_cost_function = "linear"

algorithms_labels = dict()
# algorithms_labels[k1_optimal] = dict()
# algorithms_labels[k1_optimal][k2_scheduling] = k1_optimal
# algorithms_labels[k1_optimal][k2_pricing] = "{}_fw".format(k1_optimal)
algorithms_labels[k1_heuristic] = dict()
algorithms_labels[k1_heuristic][k2_scheduling] = k1_heuristic
algorithms_labels[k1_heuristic][k2_pricing] = "{}_fw".format(k1_heuristic)


date_folder = result_folder + "{}/".format(str(date.today()))
date_time_folder = date_folder + "{}/".format(str(datetime.now().time().strftime("%H-%M-%S")))
experiment_summary_dict = dict()
for r in range(repeat_num):
    for n in household_nums:
        date_time_experiment_folder = date_time_folder \
                                      + "h{0}-t{1}-w{2}-r{3}/".format(n, no_tasks_min, care_f_weight, r)

        experiment_summary = experiment(n, no_tasks_min, no_tasks_min + 2, new_data, type_cost_function,
                                        algorithms_labels, date_time_experiment_folder)
        for algorithm in algorithms_labels.values():
            for v in algorithm.values():
                experiment_summary_dict[r, n, v] = experiment_summary[v]

        experiment_summary_pd = pd.DataFrame.from_dict(experiment_summary_dict, orient='index')
        experiment_summary_pd.reset_index().to_csv(date_time_folder + "{}.csv".format('Experiment_summary'))
