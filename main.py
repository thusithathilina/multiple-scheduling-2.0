import pickle
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import layout, column, row, grid
from bokeh.models import ColumnDataSource, RadioButtonGroup, DatePicker, Select, Div, \
    DataTable, TableColumn, NumberFormatter, Panel, Tabs, Button, HoverTool, LinearColorMapper, ColorBar, BasicTicker, \
    PrintfTickFormatter
from bokeh.plotting import figure
from os import walk, path
from pathlib import Path
from scripts.iteration import *
from datetime import date


exp_date = "2020-09-29"
exp_time = None
# parent_folder = "multiple/"
parent_folder = ""
results_folder = parent_folder + "results/"


def view_results(date_folder, time_folder):

    def dict_to_pd_dt(area_res, target_dict, k0_ks, k1_ks):
        for k0 in k0_ks:
            if k0 not in target_dict:
                target_dict[k0] = dict()
            for k1 in k1_ks:
                df = pd.DataFrame.from_dict(area_res[k0][k1], orient='index').reset_index(drop=True)
                df.columns = [str(x) for x in range(len(area_res[k0][k1][0]))]
                target_dict[k0][k1] = df
                # target_dict[k0][k1].stack().reset_index()

    def combine_dict_to_pd_dt(area_res, target_dict, k0_ks, k1_ks):
        for k1 in k1_ks:
            combined_dict = dict()
            for k0 in k0_ks:
                combined_dict[k0] = area_res[k0][k1]
            df = pd.DataFrame(combined_dict, columns=k0_ks).reset_index(drop=True)
            target_dict[k1] = df

    # ---------- initialise data trackers ----------
    def load_data(f_date, f_time):
        date_time_folder = results_folder + "{}/{}/".format(f_date, f_time)
        with open(date_time_folder + "area_output.pkl", 'rb') as f2:
            area_res = pickle.load(f2)
            f2.close()

        k0_keys = [k0_demand]
        k1_scheduling_ks = []
        k1_pricing_fw_ks = []
        all_labels = area_res[k0_demand].keys()
        for label in all_labels:
            if "fw" in label:
                k1_pricing_fw_ks.append(label)
            else:
                k1_scheduling_ks.append(label)

        k1_keys = k1_scheduling_ks
        dict_to_pd_dt(area_res, prices_dict, k0_keys, k1_keys)

        k0_keys = [k0_demand, k0_prices]
        k1_keys = k1_pricing_fw_ks
        dict_to_pd_dt(area_res, demands_prices_fw_dict, k0_keys, k1_keys)

        k0_keys = [k0_demand_max, k0_par, k0_obj, k0_cost, k0_penalty, k0_step]
        if k0_time in area_res:
            k0_keys.append(k0_time)
        if k0_demand_total in area_res:
            k0_keys.append(k0_demand_total)
        k1_keys = k1_pricing_fw_ks
        combine_dict_to_pd_dt(area_res, others_combined_dict, k0_keys, k1_keys)

        return k1_scheduling_ks, k1_pricing_fw_ks

    prices_dict = dict()
    demands_prices_fw_dict = dict()
    others_combined_dict = dict()
    if date_folder is None:
        date_folder = str(date.today())
    if time_folder is None:
        time_folder = [dirs for root, dirs, _ in walk(results_folder + date_folder) if dirs != []][0][0]
    k1_scheduling_keys, k1_pricing_fw_keys = load_data(date_folder, time_folder)

    # ------------------------------ 1. Data Tab ------------------------------ #
    # 1.1. initialise web widgets
    # 1.1.1 - date picker: choose the date of the results
    date_default = str(date.today()) if date_folder is None else date_folder
    # date_options = [dirs for root, dirs, _ in walk(results_folder) if dirs != []][0]
    date_min = "2020-09-27"
    date_max = str(date.today())
    date_picker = DatePicker(title='Select date:', value=date_default, min_date=date_min, max_date=date_max)

    # 1.1.2 - select: choose the time of the results
    select_time = Select(title="Select time:", value=time_folder)

    # 1.1.3 - select: choose the algorithm
    chosen_algorithm = [k for k in k1_pricing_fw_keys if "optimal" in k][0]
    algorithm_options = k1_pricing_fw_keys
    select_algorithm = Select(title="Select algorithm:", value=chosen_algorithm, options=algorithm_options)

    # 1.1.4 - div: show the experiment summary
    div = Div()

    # 1.1.5 - radio buttons: choose the date source
    radio_button_group = RadioButtonGroup(labels=["Statistics", "Demand Profile", "Price Profile"], active=0)

    # 1.1.6 - data table: show numbers
    source = ColumnDataSource()
    data_table = DataTable(source=source, index_position=None)

    # 1.2. design layout
    layout_data = layout([
        [radio_button_group],
        [data_table]

    ], sizing_mode='scale_width')
    tab_data = Panel(child=layout_data, title='Data')

    # ------------------------------ 2. Graph Tab ------------------------------ #
    # 2.1. data source
    k1_algorithm = select_algorithm.value
    # source_combined = ColumnDataSource(others_combined_dict[k1_algorithm])
    source_combined = ColumnDataSource()
    hover = HoverTool(tooltips=[('Iteration', '@index'), ('Cost', '@cost'), ('Max demand', '@max_demand')])

    def draw_bar_chart(source_data, title, x_label, y_label, colour, x_data, top_data):
        p = figure(title=title, background_fill_color="#fafafa", plot_height=350,
                   x_axis_label=x_label, y_axis_label=y_label)
        p.line(x_data, top_data, source=source_data)
        p.circle(x_data, top_data, size=5, source=source, selection_color="orange")
        p.add_tools(hover)
        return p

    # 2.2. cost bar chart
    p_cost = draw_bar_chart(source_data=source_combined, title='Cost per Iteration',
                            x_label='Iteration', y_label='Cost (cent)', colour='orange',
                            x_data='index', top_data=k0_cost)

    # 2.3 max demand bar chart
    p_demand_max = draw_bar_chart(source_data=source_combined, title='Max demand per Iteration',
                                  x_label='Iteration', y_label='Demand (KW)', colour='green',
                                  x_data='index', top_data=k0_demand_max)

    def draw_demand_price_heatmap(dtype, x_loc, colors, k1_alg):
        data = demands_prices_fw_dict[dtype][k1_alg]

        x_periods = list(data.columns)
        y_iterations = [str(x) for x in (list(data.index))]
        data = data.iloc[::-1].stack().reset_index()
        data.columns = ['Iteration', 'Period', dtype]
        source_heatmap = ColumnDataSource(data)
        tooltips = [('Iteration', '@Iteration'), ('Period', '@Period'), ('Value', '@' + dtype)]

        mapper = LinearColorMapper(palette=colors, low=data[dtype].min(), high=data[dtype].max())
        p = figure(title='{} Heatmap'.format(dtype), x_range=x_periods, y_range=y_iterations,
                         x_axis_location=x_loc, tooltips=tooltips, plot_width=900, plot_height=350)

        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "7px"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = pi / 3

        p.rect(x="Period", y="Iteration", width=1, height=1,
               source=source_heatmap,
               fill_color={'field': dtype, 'transform': mapper},
               line_color=None)
        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7px",
                             ticker=BasicTicker(desired_num_ticks=len(colors)),
                             formatter=PrintfTickFormatter(format="%d%%"),
                             label_standoff=6, border_line_color=None, location=(0, 0))
        p.add_layout(color_bar, 'right')

        return p, source_heatmap

    # 2.2 demand heatmap and price heatmap
    x_location = "above"
    heatmap_colours = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]

    data_type = k0_demand
    heatmap_demand, source_heatmap_demand = draw_demand_price_heatmap(data_type, x_location, heatmap_colours, k1_algorithm)

    data_type = k0_prices
    heatmap_price, source_heatmap_price = draw_demand_price_heatmap(data_type, x_location, heatmap_colours, k1_algorithm)

    # 2.4 graph tab layout
    row1 = row(p_cost, heatmap_demand)
    row2 = row(p_demand_max, heatmap_price)
    # todo - stretch is not working yet. want to make it responsive
    layout_graph = layout(column([row1, row2]), sizing_mode='stretch_both')
    tab_graph = Panel(child=layout_graph, title='Graph')

    # ------------------------------ 3. Input Tab ------------------------------ #
    def run_experiment():
        # iteration(100, no_tasks, True)
        print("TBC")

    # input the parameters for running the experiments

    # a run button

    # a text window for streaming the results from the python program (not sure if I need that)
    button = Button(label="Run", button_type="primary")
    button.on_click(run_experiment)

    layout_exp = layout([button], sizing_mode='scale_width')
    tab_exp = Panel(child=layout_exp, title='Experiment')

    # ------------------------------ 4. event functions for widgets ------------------------------ #
    def update_select_options(attr, old, new):
        select_opt = [dirs for root, dirs, _ in walk(results_folder + "{}".format(new)) if dirs != []][0]
        select_time.options = select_opt
        select_time.value = select_opt[-1]

        # todo - will try to sort the folders later again
        # selected_date_folder = Path(results_folder + "{}".format(selected_date))
        # select_options = sorted(selected_date_folder.iterdir(), key=path.getmtime, reverse=True)
        # select_options = [x.parts[-1] for x in select_options if x.is_dir()]

    def update_data_table_content_and_graph(attr, old, new):
        # print(select_algorithm.value)
        if new == 1:  # demand
            source.data = demands_prices_fw_dict[k0_demand][select_algorithm.value]
        if new == 2:  # prices
            source.data = demands_prices_fw_dict[k0_prices][select_algorithm.value]
        if new == 0:  # others
            # display_keys = list(source.data.keys())
            source.data = others_combined_dict[select_algorithm.value]
        # print(source.data)
        columns_keys = source.data.keys()
        table_columns = [TableColumn(field=str(i), title=str(i).replace("_", " ").capitalize(),
                                     formatter=NumberFormatter(format="0,0.00", text_align="right"))
                         for i in columns_keys]
        data_table.columns = table_columns
        data_table.update()

        source_combined.data = others_combined_dict[select_algorithm.value]
        p_cost.update()
        p_demand_max.update()

        # data = demands_prices_fw_dict[k0_demand][select_algorithm.value]
        # data = data.iloc[::-1].stack().reset_index()
        # data.columns = ['Iteration', 'Period', k0_demand]
        # source_heatmap_demand.data = data
        # mapper = LinearColorMapper(palette=heatmap_colours, low=data[k0_demand].min(), high=data[k0_demand].max())
        # heatmap_demand.rect(x="Period", y="Iteration", width=1, height=1,
        #                     source=source_heatmap_demand,
        #                     fill_color={'field': k0_demand, 'transform': mapper},
        #                     line_color=None)
        #
        # data = demands_prices_fw_dict[k0_prices][select_algorithm.value]
        # data = data.iloc[::-1].stack().reset_index()
        # data.columns = ['Iteration', 'Period', k0_prices]
        # source_heatmap_demand.data = data
        # source_heatmap_price.data = demands_prices_fw_dict[k0_prices][select_algorithm.value]
        # mapper = LinearColorMapper(palette=heatmap_colours, low=data[k0_prices].min(), high=data[k0_prices].max())
        # heatmap_price.rect(x="Period", y="Iteration", width=1, height=1,
        #                    source=source_heatmap_price,
        #                    fill_color={'field': k0_prices, 'transform': mapper},
        #                    line_color=None)

        # heatmap_demand.update()
        # heatmap_price.update()

    def update_div_content(attr, d_f, t_f):
        # d_f = date_picker.value
        # t_f = select.value
        f = open(results_folder + "{}/{}/".format(d_f, t_f) + 'summary.txt', 'r+')
        str_summary = f.read()
        f.close()
        div.text = str_summary

    def update_data_table_source(attr, old, new):
        d_folder = date_picker.value
        t_folder = new
        load_data(d_folder, t_folder)
        active_radio_button = radio_button_group.active
        update_data_table_content_and_graph(None, None, active_radio_button)
        update_div_content(None, d_folder, t_folder)

    def switch_algorithm(attr, old, new):
        active_radio_button = radio_button_group.active
        update_data_table_content_and_graph(None, None, active_radio_button)

    update_select_options(None, None, date_picker.value)
    update_data_table_content_and_graph(None, None, 0)
    update_div_content(None, date_picker.value, select_time.value)

    # ------------------------------ 5. assign event functions to widgets ------------------------------ #
    date_picker.on_change("value", update_select_options)
    select_time.on_change("value", update_data_table_source)
    select_algorithm.on_change("value", switch_algorithm)
    radio_button_group.on_change("active", update_data_table_content_and_graph)

    # ------------------------------ 4. Show all ------------------------------ #
    tabs = Tabs(tabs=[tab_data, tab_graph], sizing_mode='scale_both')
    layout_overall = layout([
        [date_picker, select_time, select_algorithm, div],
        tabs
    ], sizing_mode='scale_width')

    curdoc().add_root(layout_overall)


view_results(exp_date, exp_time)


