import pickle
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import layout, column, row, grid
from bokeh.models import ColumnDataSource, RadioButtonGroup, DatePicker, Select, Div, \
    DataTable, TableColumn, NumberFormatter, Panel, Tabs, Range1d, HoverTool, LinearColorMapper, ColorBar, BasicTicker, \
    PrintfTickFormatter
from bokeh.plotting import figure
from os import walk, path
from pathlib import Path
from scripts.input_parameter import *
from datetime import date
from math import pi

exp_date = "2020-10-04"
# exp_date = None
exp_time = "04-29-41-234378"
# parent_folder = "multiple/"
parent_folder = ""
results_folder = parent_folder + "results/"


def to_pd(sum_dt, target_dict):
    for k0, v0 in sum_dt.items():
        for k1, v1 in v0.items():
            if k1 not in target_dict:
                target_dict[k1] = dict()
            target_dict[k1][k0] = v1


def dict_to_pd_dt(area_res, target_dict, k0_ks, k1_ks):
    for k0 in k0_ks:
        if k0 not in target_dict:
            target_dict[k0] = dict()
        for k1 in k1_ks:
            df = pd.DataFrame.from_dict(area_res[k0][k1], orient='index').reset_index(drop=True)
            df.columns = [str(x) for x in range(len(area_res[k0][k1][0]))]
            target_dict[k0][k1] = df
            # target_dict[k0][k1].stack().reset_index()


def combine_dict_to_pd_dt(area_res, target_dict, k0_ks, k1_s, k1_p):
    for ks, kp in zip(k1_s, k1_p):
        combined_dict = dict()
        pd_columns = []
        for k0 in k0_ks:
            if k0 == k0_time:
                combined_dict[k1_time_scheduling] = area_res[k0][ks]
                combined_dict[k1_time_pricing] = area_res[k0][kp]
                pd_columns.extend([k1_time_scheduling, k1_time_pricing])
            else:
                combined_dict[k0] = area_res[k0][kp]
                pd_columns.append(k0)
        df = pd.DataFrame(combined_dict, columns=pd_columns).reset_index(drop=True)
        target_dict[kp] = df


def draw_line_chart(source_data, title, x_label, y_label, colour, x_data, top_data, hover):
    p = figure(title=title, background_fill_color="#fafafa", plot_height=350,
               x_axis_label=x_label, y_axis_label=y_label)
    p.y_range.start = 0
    p.line(x_data, top_data, source=source_data)
    p.circle(x_data, top_data, size=5, source=source_data, selection_color="orange")
    p.add_tools(hover)
    return p


def draw_demand_price_heatmap(source_heatmap, dtype, x_loc, colors):
    tooltips = [('Iteration', '@Iteration'), ('Period', '@Period'), (dtype.capitalize(), '@' + dtype)]

    mapper = LinearColorMapper(palette=colors, low=0, high=999999)
    p = figure(title='{} Heatmap'.format(dtype.capitalize()),
               y_range=[str(x) for x in range(10)],
               x_range=[str(x) for x in range(48)],
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

    return p, chart, color_bar, mapper


def make_overview_tab():
    # labels = ["Summary", "Statistics", "Demand Profile", "Price Profile"]
    labels = ["Summary", "Statistics"]
    radio_button_group = RadioButtonGroup(labels=labels, active=0)
    source_datatable = ColumnDataSource()
    data_table = DataTable(source=source_datatable, index_position=None)

    return radio_button_group, data_table, source_datatable


def make_graph_tab(s_combined, s_heatmap_demand, s_heatmap_price):
    # 4.1 line charts
    hover = HoverTool(tooltips=[('Iteration', '@index'), ('Cost', '@cost'), ('Max demand', '@max_demand')])
    plot_line_cost = draw_line_chart(source_data=s_combined, title='Cost per Iteration',
                                     x_label='Iteration', y_label='Cost (cent)', colour='orange',
                                     x_data='index', top_data=k0_cost, hover=hover)
    plot_line_demand_max = draw_line_chart(source_data=s_combined, title='Max demand per Iteration',
                                           x_label='Iteration', y_label='Demand (KW)', colour='green',
                                           x_data='index', top_data=k0_demand_max, hover=hover)

    # 4.2 heatmaps
    x_location = "above"
    heatmap_colours = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
                       "#550b1d"]
    plot_heatmap_demand, chart_heatmap_demand, color_bar_demand, mapper_demand \
        = draw_demand_price_heatmap(s_heatmap_demand, k0_demand, x_location, heatmap_colours)

    plot_heatmap_price, chart_heatmap_price, color_bar_prices, mapper_price \
        = draw_demand_price_heatmap(s_heatmap_price, k0_prices, x_location, heatmap_colours)

    return {"line": {k0_cost: plot_line_cost, k0_demand_max: plot_line_demand_max},
            "heatmap": {"plot": {k0_demand: plot_heatmap_demand, k0_prices: plot_heatmap_price},
                        "chart": {k0_demand: chart_heatmap_demand, k0_prices: chart_heatmap_price},
                        "colour": {k0_demand: color_bar_demand, k0_prices: color_bar_prices},
                        "mapper": {k0_demand: mapper_demand, k0_prices: mapper_price}
                        }
            }


def make_header(date_folder, time_folder, res_folder):
    # 1.1 - date picker: choose the date of the results
    date_default = str(date.today()) if date_folder is None else date_folder
    date_available = [dirs for root, dirs, _ in walk(res_folder) if dirs != []][0]
    date_picker = DatePicker(title='Select date:', value=date_default,
                             min_date="2020-09-27", max_date=str(date.today()), enabled_dates=date_available)

    # 1.2 - select: choose the time of the results
    select_time = Select(title="Select time:", value=time_folder)

    # 1.3 - select: choose the algorithm
    select_algorithm = Select(title="Select algorithm:")

    # 1.1.4 - div: show the experiment summary
    div = Div()

    return date_picker, select_time, select_algorithm, div


# def make_demand_price_tab(s_demands, s_prices):
#     x_label = [str(i) for i in range(48)]
#     # y_label = [str(i) for i in range(10)]
#     p_demand = figure(title="Demand Profiles", background_fill_color="#fafafa", plot_height=600,
#                       x_axis_label=x_label, y_axis_label=y_label)
#     p_demand.line(x_data, top_data, source=source_data)
#     p.circle(x_data, top_data, size=5, source=source_data, selection_color="orange")
#     p.add_tools(hover)
#
#
#     p_prices = figure(title="Prices Profiles", background_fill_color="#fafafa", plot_height=600,
#                       x_axis_label=x_label, y_axis_label=y_label)
#
#     return p_demand, p_prices


def view_results(date_folder, time_folder, res_folder):
    # ------------------------------ 1. draw widgets ------------------------------ #
    # 1.1 header
    header_date, header_time, header_algorithm, header_note \
        = make_header(date_folder, time_folder, res_folder)
    header_row = row(header_date, header_time, header_algorithm, header_note, sizing_mode='scale_width')

    # 1.2 summary tab

    # 1.3 overview Tab
    data_radio_button_group, data_table, source_datatable = make_overview_tab()
    layout_data = layout([
        [data_radio_button_group],
        [data_table]
    ], sizing_mode='scale_width')
    tab_data = Panel(child=layout_data, title='Overview')

    # 1.4 cost, max demand, prices and demands tab
    source_combined = ColumnDataSource()
    source_heatmap_demand = ColumnDataSource()
    source_heatmap_price = ColumnDataSource()
    graph_dict = make_graph_tab(source_combined, source_heatmap_demand, source_heatmap_price)
    plot_line_cost = graph_dict["line"][k0_cost]
    plot_line_demand_max = graph_dict["line"][k0_demand_max]

    plot_heatmap_demand = graph_dict["heatmap"]["plot"][k0_demand]
    chart_heatmap_demand = graph_dict["heatmap"]["chart"][k0_demand]
    color_bar_demand = graph_dict["heatmap"]["colour"][k0_demand]
    mapper_demand = graph_dict["heatmap"]["mapper"][k0_demand]

    plot_heatmap_price = graph_dict["heatmap"]["plot"][k0_prices]
    chart_heatmap_price = graph_dict["heatmap"]["chart"][k0_prices]
    color_bar_prices = graph_dict["heatmap"]["colour"][k0_prices]
    mapper_price = graph_dict["heatmap"]["mapper"][k0_prices]

    # todo - stretch is not working yet. want to make it responsive
    row1 = row(plot_line_cost, plot_heatmap_demand)
    row2 = row(plot_line_demand_max, plot_heatmap_price)
    layout_graph = layout(column([row1, row2]), sizing_mode='stretch_both')
    tab_graph = Panel(child=layout_graph, title='Cost, Max demand, Demands and Prices')

    # 1.5 demands and prices line charts
    # source_line_demands = ColumnDataSource()
    # source_line_prices = ColumnDataSource()
    # plot_demands, plot_prices = make_demand_price_tab(source_line_demands, source_line_prices)
    # layout_demands_prices = layout(row(plot_demands, plot_prices, sizing_mode='scale_both'))
    # tab_demands_prices = Panel(child=layout_demands_prices, title='Demands and Prices')

    # 1.6 overall layout all
    tabs = Tabs(tabs=[tab_data, tab_graph], sizing_mode='scale_both')
    layout_overall = layout([
        header_row,
        tabs
    ], sizing_mode='scale_width')

    # ------------------------------ 2. event functions for widgets ------------------------------ #
    prices_dict = dict()
    demands_prices_fw_dict = dict()
    others_combined_dict = dict()
    summary_dict = dict()

    def update_heatmap(chosen_algorithm, k0_label, source, plot, mapper, chart, colour_bar):
        data = demands_prices_fw_dict[k0_label][chosen_algorithm]
        x_periods = [str(x) for x in (list(data.columns))]
        y_iterations = [str(x) for x in (list(data.index))]
        plot.y_range.factors = y_iterations
        plot.x_range.factors = x_periods

        data = data.iloc[::-1].stack().reset_index()
        data.columns = ['Iteration', 'Period', k0_label]
        source.data = data

        mapper.low = data[k0_label].min()
        mapper.high = data[k0_label].max()

        chart.update()
        colour_bar.update()

    def update_line_chart(chosen_algorithm):
        source_combined.data = others_combined_dict[chosen_algorithm]
        # plot_line_cost.y_range = Range1d(0, int(source_combined.data[k0_cost].max() * 1.1))
        plot_line_cost.update()

        # plot_line_demand_max.y_range = Range1d(0, int(source_combined.data[k0_demand_max].max() + 1))
        plot_line_demand_max.update()

    def update_data_table(active_radio_button, chosen_algorithm):
        if active_radio_button == 0:  # summary
            source_datatable.data = pd.DataFrame.from_dict(summary_dict, orient='index')
        elif active_radio_button == 1:  # statistics
            source_datatable.data = others_combined_dict[chosen_algorithm]
        elif active_radio_button == 2:  # demand
            source_datatable.data = demands_prices_fw_dict[k0_demand][chosen_algorithm]
        elif active_radio_button == 3:  # prices
            source_datatable.data = demands_prices_fw_dict[k0_prices][chosen_algorithm]

        columns_keys = source_datatable.data.keys()
        if active_radio_button == 0:
            table_columns = [TableColumn(field=str(i), title=str(i).replace("_", " ").capitalize())
                             for i in columns_keys]
        else:
            table_columns = [TableColumn(field=str(i), title=str(i).replace("_", " ").capitalize(),
                                         formatter=NumberFormatter(format="0,0.00", text_align="right"))
                             for i in columns_keys]
        data_table.columns = table_columns
        data_table.update()

    def update_header_algorithm(keys):
        chosen_algorithm = [k for k in keys if "optimal" in k][0]
        header_algorithm.value = chosen_algorithm
        header_algorithm.options = keys
        header_algorithm.update()

    def update_data_source(new_time):
        date_t = header_date.value
        time_t = new_time
        date_time_folder = results_folder + "{}/{}/".format(date_t, time_t)

        f = open(date_time_folder + 'note.txt', 'r+')
        str_summary = f.read()
        f.close()
        header_note.text = str_summary

        with open(date_time_folder + "summary.pkl", 'rb') as f2:
            summary = pickle.load(f2)
            f2.close()
        to_pd(summary, summary_dict)

        # df.columns = [str(x) for x in range(len(area_res[k0][k1][0]))]

        with open(date_time_folder + "area_output.pkl", 'rb') as f2:
            area_res = pickle.load(f2)
            f2.close()
        k1_scheduling_ks = []
        k1_pricing_fw_ks = []
        all_labels = area_res[k0_demand].keys()
        for label in all_labels:
            if "fw" in label:
                k1_pricing_fw_ks.append(label)
            else:
                k1_scheduling_ks.append(label)

        dict_to_pd_dt(area_res, prices_dict, [k0_demand], k1_scheduling_ks)
        dict_to_pd_dt(area_res, demands_prices_fw_dict, [k0_demand, k0_prices], k1_pricing_fw_ks)

        k0_keys = [k0_demand_max, k0_par, k0_obj, k0_cost, k0_penalty, k0_step]
        if k0_time in area_res:
            k0_keys.append(k0_time)
        if k0_demand_total in area_res:
            k0_keys.append(k0_demand_total)
        combine_dict_to_pd_dt(area_res, others_combined_dict, k0_keys, k1_scheduling_ks, k1_pricing_fw_ks)

        return k1_scheduling_ks, k1_pricing_fw_ks

    def callback_update_data_table(attr, old, active_radio_button):
        update_data_table(active_radio_button, header_algorithm.value)

    def callback_update_data_source(attr, old, new_time):
        # read new data source
        k1_scheduling_keys, k1_pricing_fw_keys = update_data_source(new_time)

        # update the algorithm selection
        update_header_algorithm(k1_pricing_fw_keys)
        chosen_algorithm = header_algorithm.value
        callback_switch_algorithm(None, None, chosen_algorithm)

    def callback_update_header_time_options(attr, old, new_date):
        select_opt = [dirs for root, dirs, _ in walk(results_folder + "{}".format(new_date)) if dirs != []][0]
        header_time.options = select_opt
        # todo - choose the latest dataset
        header_time.value = select_opt[-1]

    def callback_switch_algorithm(attr, old, chosen_algorithm):
        # update the datatable content
        active_button = data_radio_button_group.active
        update_data_table(active_button, chosen_algorithm)

        # update the line graphs
        update_line_chart(chosen_algorithm)

        # update the heat maps
        update_heatmap(chosen_algorithm=chosen_algorithm, k0_label=k0_demand, source=source_heatmap_demand,
                       plot=plot_heatmap_demand, mapper=mapper_demand, chart=chart_heatmap_demand,
                       colour_bar=color_bar_demand)
        update_heatmap(chosen_algorithm=chosen_algorithm, k0_label=k0_prices, source=source_heatmap_price,
                       plot=plot_heatmap_price, mapper=mapper_price, chart=chart_heatmap_price,
                       colour_bar=color_bar_prices)
        # update_heatmap(k0_demand, source_heatmap_demand, plot_heatmap_demand, mapper_demand,
        #                chart_heatmap_demand, color_bar_demand)
        # update_heatmap(k0_prices, source_heatmap_price, plot_heatmap_price, mapper_price,
        #                chart_heatmap_price, color_bar_prices)

    # ------------------------------ 3. assign event functions to widgets ------------------------------ #
    header_date.on_change("value", callback_update_header_time_options)
    header_time.on_change("value", callback_update_data_source)
    header_algorithm.on_change("value", callback_switch_algorithm)
    data_radio_button_group.on_change("active", callback_update_data_table)

    curdoc().add_root(layout_overall)

    # ------------------------------ 4. initialise ------------------------------ #
    start_date = exp_date if exp_date is not None else str(date.today())
    start_time = exp_time if exp_time is not None else header_time.value

    callback_update_header_time_options(None, None, start_date)
    callback_update_data_source(None, None, start_time)
    callback_switch_algorithm(None, None, header_algorithm.value)
    # callback_update_data_table(None, None, 0)


view_results(exp_date, exp_time, results_folder)
