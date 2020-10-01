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

exp_date = "2020-10-01"
exp_time = "01-22-14-924742"
# parent_folder = "multiple/"
parent_folder = ""
results_folder = parent_folder + "results/"


def dict_to_pd_dt(area_res, target_dict, k0_ks, k1_ks):
    for k0 in k0_ks:
        if k0 not in target_dict:
            target_dict[k0] = dict()
        for k1 in k1_ks:
            df = pd.DataFrame.from_dict(area_res[k0][k1], orient='index').reset_index(drop=True)
            df.columns = [str(x) for x in range(len(area_res[k0][k1][0]))]
            target_dict[k0][k1] = df
            # target_dict[k0][k1].stack().reset_index()
    return target_dict


def combine_dict_to_pd_dt(area_res, target_dict, k0_ks, k1_ks):
    for k1 in k1_ks:
        combined_dict = dict()
        for k0 in k0_ks:
            combined_dict[k0] = area_res[k0][k1]
        df = pd.DataFrame(combined_dict, columns=k0_ks).reset_index(drop=True)
        target_dict[k1] = df
    return target_dict


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

    prices_dt = dict()
    prices_dt = dict_to_pd_dt(area_res, prices_dt, k0_keys, k1_scheduling_ks)

    k0_keys = [k0_demand, k0_prices]
    demands_prices_fw_dt = dict()
    demands_prices_fw_dt = dict_to_pd_dt(area_res, demands_prices_fw_dt, k0_keys, k1_pricing_fw_ks)

    k0_keys = [k0_demand_max, k0_par, k0_obj, k0_cost, k0_penalty, k0_step]
    if k0_time in area_res:
        k0_keys.append(k0_time)
    if k0_demand_total in area_res:
        k0_keys.append(k0_demand_total)
    others_combined_dt = dict()
    others_combined_dt = combine_dict_to_pd_dt(area_res, others_combined_dt, k0_keys, k1_pricing_fw_ks)

    return prices_dt, demands_prices_fw_dt, others_combined_dt, k1_scheduling_ks, k1_pricing_fw_ks


def draw_line_chart(source_data, title, x_label, y_label, colour, x_data, top_data, hover):
    p = figure(title=title, background_fill_color="#fafafa", plot_height=350,
               x_axis_label=x_label, y_axis_label=y_label)
    p.line(x_data, top_data, source=source_data)
    p.circle(x_data, top_data, size=5, source=source_data, selection_color="orange")
    p.add_tools(hover)
    return p


def draw_demand_price_heatmap(data_dict, dtype, x_loc, colors, k1_alg):
    data = data_dict[dtype][k1_alg]

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

    chart = p.rect(x="Period", y="Iteration", width=1, height=1,
                   source=source_heatmap,
                   fill_color={'field': dtype, 'transform': mapper},
                   line_color=None)
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7px",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d%%"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    return p, source_heatmap, chart, color_bar, mapper


def make_data_tab():
    radio_button_group = RadioButtonGroup(labels=["Statistics", "Demand Profile", "Price Profile"], active=0)
    data_table = DataTable(index_position=None)

    return radio_button_group, data_table


def make_graph_tab():
    True


def make_header(date_folder, time_folder, res_folder, k1_keys):
    # 1.1 - date picker: choose the date of the results
    date_default = str(date.today()) if date_folder is None else date_folder
    date_available = [dirs for root, dirs, _ in walk(res_folder) if dirs != []][0]
    date_picker = DatePicker(title='Select date:', value=date_default,
                             min_date="2020-09-27", max_date=str(date.today()), enabled_dates=date_available)

    # 1.2 - select: choose the time of the results
    select_time = Select(title="Select time:", value=time_folder)

    # 1.3 - select: choose the algorithm
    chosen_algorithm = [k for k in k1_keys if "optimal" in k][0]
    select_algorithm = Select(title="Select algorithm:", value=chosen_algorithm, options=k1_keys)

    # 1.1.4 - div: show the experiment summary
    div = Div()

    return date_picker, select_time, select_algorithm, div


def view_results(date_folder, time_folder, res_folder):

    # 0. load data
    if date_folder is None:
        date_folder = str(date.today())
    if time_folder is None:
        time_folder = [dirs for root, dirs, _ in walk(results_folder + date_folder) if dirs != []][0][0]
    prices_dict, demands_prices_fw_dict, others_combined_dict, k1_scheduling_keys, k1_pricing_fw_keys \
        = load_data(date_folder, time_folder)

    # 1. header
    header_date, header_time, header_algorithm, header_note \
        = make_header(date_folder, time_folder, res_folder, k1_pricing_fw_keys)

    # 2. summary tab

    # 3. data Tab
    data_radio_button_group, data_table = make_data_tab()
    source_datatable = ColumnDataSource()
    data_table.source = source_datatable

    # 3.3. design layout
    layout_data = layout([
        [data_radio_button_group],
        [data_table]
    ], sizing_mode='scale_width')
    tab_data = Panel(child=layout_data, title='Data')

    # ------------------------------ 4. Graph Tab ------------------------------ #
    # 4.1. data source
    source_combined = ColumnDataSource()
    source_heatmap_demand = ColumnDataSource()
    source_heatmap_price = ColumnDataSource()

    # 4.2. line charts
    plot_line_cost = figure()
    plot_line_demand_max = figure()

    # 4.3 heatmaps
    plot_heatmap_demand = figure()
    plot_heatmap_price = figure()


    # source_combined = ColumnDataSource()
    # hover = HoverTool(tooltips=[('Iteration', '@index'), ('Cost', '@cost'), ('Max demand', '@max_demand')])
    # p_cost = draw_line_chart(source_data=source_combined, title='Cost per Iteration',
    #                          x_label='Iteration', y_label='Cost (cent)', colour='orange',
    #                          x_data='index', top_data=k0_cost, hover=hover)
    # p_demand_max = draw_line_chart(source_data=source_combined, title='Max demand per Iteration',
    #                                x_label='Iteration', y_label='Demand (KW)', colour='green',
    #                                x_data='index', top_data=k0_demand_max, hover=hover)
    # x_location = "above"
    # heatmap_colours = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
    #                    "#550b1d"]
    # k1_algorithm = select_algorithm.value
    # plot_heatmap_demand, source_heatmap_demand, chart_heatmap_demand, color_bar_demand, mapper_demand \
    #     = draw_demand_price_heatmap(demands_prices_fw_dict, k0_demand, x_location, heatmap_colours, k1_algorithm)
    #
    # plot_heatmap_price, source_heatmap_price, chart_heatmap_price, color_bar_prices, mapper_price \
    #     = draw_demand_price_heatmap(demands_prices_fw_dict, k0_prices, x_location, heatmap_colours, k1_algorithm)

    # 4.5 graph tab layout
    row1 = row(plot_line_cost, plot_heatmap_demand)
    row2 = row(plot_line_demand_max, plot_heatmap_price)
    # todo - stretch is not working yet. want to make it responsive
    layout_graph = layout(column([row1, row2]), sizing_mode='stretch_both')
    tab_graph = Panel(child=layout_graph, title='Graph')

    # ------------------------------ 4. event functions for widgets ------------------------------ #
    def update_select_options(attr, old, new):
        select_opt = [dirs for root, dirs, _ in walk(results_folder + "{}".format(new)) if dirs != []][0]
        select_time.options = select_opt
        select_time.value = select_opt[-1]

    def update_data_table(active_radio_button):

        if active_radio_button == 1:  # demand
            source_datatable.data = demands_prices_fw_dict[k0_demand][select_algorithm.value]
        if active_radio_button == 2:  # prices
            source_datatable.data = demands_prices_fw_dict[k0_prices][select_algorithm.value]
        if active_radio_button == 0:  # others
            source_datatable.data = others_combined_dict[select_algorithm.value]

        columns_keys = source_datatable.data.keys()
        table_columns = [TableColumn(field=str(i), title=str(i).replace("_", " ").capitalize(),
                                     formatter=NumberFormatter(format="0,0.00", text_align="right"))
                         for i in columns_keys]
        data_table.columns = table_columns
        data_table.update()

    def update_line_chart():
        source_combined.data = others_combined_dict[select_algorithm.value]
        plot_cost.update()
        plot_demand_max.update()

    def update_heatmap(k0_label, source, plot, mapper, chart, colour_bar):
        data = demands_prices_fw_dict[k0_label][select_algorithm.value]
        y_iterations = [str(x) for x in (list(data.index))]
        plot.y_range.factors = y_iterations

        data = data.iloc[::-1].stack().reset_index()
        data.columns = ['Iteration', 'Period', k0_label]
        source.data = data

        mapper.low = data[k0_label].min()
        mapper.high = data[k0_label].max()

        chart.update()
        colour_bar.update()

    def update_content(attr, old, new):

        update_data_table(new)
        update_line_chart()

        update_heatmap(k0_demand, source_heatmap_demand, plot_heatmap_demand, mapper_demand,
                       chart_heatmap_demand, color_bar_demand)
        update_heatmap(k0_prices, source_heatmap_price, plot_heatmap_price, mapper_price,
                       chart_heatmap_price, color_bar_prices)

    def update_div_content(attr, d_f, t_f):
        f = open(results_folder + "{}/{}/".format(d_f, t_f) + 'summary.txt', 'r+')
        str_summary = f.read()
        f.close()
        div.text = str_summary

    def update_source(attr, old, new):
        d_folder = date_picker.value
        t_folder = new
        prices_dict, demands_prices_fw_dict, others_combined_dict, k1_scheduling_keys, k1_pricing_fw_keys \
            = load_data(date_folder, time_folder, prices_dict, demands_prices_fw_dict, others_combined_dict)
        active_radio_button = radio_button_group.active
        update_content(None, None, active_radio_button)
        update_div_content(None, d_folder, t_folder)

    def switch_algorithm(attr, old, new):
        active_radio_button = radio_button_group.active
        update_content(None, None, active_radio_button)

    prices_dict, demands_prices_fw_dict, others_combined_dict, k1_scheduling_keys, k1_pricing_fw_keys \
        = load_data(date_folder, time_folder, prices_dict, demands_prices_fw_dict, others_combined_dict)
    update_select_options(None, None, date_picker.value)
    update_content(None, None, 0)
    update_div_content(None, date_picker.value, select_time.value)

    # ------------------------------ 5. assign event functions to widgets ------------------------------ #
    date_picker.on_change("value", update_select_options)
    select_time.on_change("value", update_source)
    select_algorithm.on_change("value", switch_algorithm)
    radio_button_group.on_change("active", update_content)

    # ------------------------------ 4. Show all ------------------------------ #
    tabs = Tabs(tabs=[tab_data, tab_graph], sizing_mode='scale_both')
    layout_overall = layout([
        [date_picker, select_time, select_algorithm, div],
        tabs
    ], sizing_mode='scale_width')

    curdoc().add_root(layout_overall)


view_results(exp_date, exp_time, results_folder)
