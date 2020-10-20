from scripts.experiment import *
from scripts.output_results import write_batch_experiment_summary
from datetime import date, datetime
from scripts.data_mode import DataMode

repeat_num = 1
enable_fdia = False
injection_percentage = 1/100
# household_nums = [2000, 4000, 6000, 8000, 10000]
# household_nums = [20, 40, 60, 80, 100]
# household_nums = household_nums.reverse()
household_nums = [10]
data_mode = DataMode.CSV_CONVERT
job_file = 'data/2017-06-01_avg.csv'
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

this_date = str(date.today())
this_time = str(datetime.now().time().strftime("%H-%M-%S"))
date_folder = result_folder + "{}/".format(this_date)
date_time_folder = date_folder + "{}/".format(this_time)
attack_result_file_prepend = this_date + "_" + this_time

experiment_summary_dict = dict()
group_by_columns = [k0_households_no, k0_tasks_no, "algorithm", k0_penalty_weight, k0_cost_type]


def run():
    for n in household_nums:
        for r in range(repeat_num):
            date_time_experiment_folder = date_time_folder \
                                          + "h{0}-t{1}-w{2}-r{3}/".format(n, no_tasks_min, care_f_weight, r)

            experiment_summary = experiment(n, no_tasks_min, no_tasks_min + 2, data_mode, type_cost_function,
                                            algorithms_labels, date_time_experiment_folder,
                                            job_file, enable_fdia, injection_percentage, attack_result_file_prepend)
            for algorithm in algorithms_labels.values():
                for v in algorithm.values():
                    experiment_summary_dict[r, n, v] = experiment_summary[v]

            # write batch experiment summary
            write_batch_experiment_summary(experiment_summary_dict, group_by_columns, date_time_folder, this_time)


if __name__ == '__main__':
    run()
