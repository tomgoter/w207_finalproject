# -*- coding: utf-8 -*-
"""
This script is used to generate a bokeh dashboard of results from
the Summer 2019, Applied Machine Learning Final Project

Thomas Goter

July 26, 2019
"""

import pandas as pd
import os
from bokeh.layouts import row, column, layout
from bokeh.models import Select
import bokeh.palettes as pal
from bokeh.plotting import curdoc, figure
from bokeh.transform import linear_cmap
from bokeh.models import ColumnDataSource, Div
from bokeh.models.widgets import Slider, Select, TextInput, DataTable, TableColumn, NumberFormatter

agg_df = pd.read_pickle('OutputData/aggregated_cnn_df.pkl')

          
agg_df.columns = ['Architecture', 'Optimizer', 'Initial Dropout Rate', 'Dropout Step',
                  'Starting Filter Depth', 'Bias Term', 'Batch Normalization', 'Stride', 
                  'Learning Rate', 'Flipped', 'Fully Connected Layer 1 Width', 'Fully Connected Layer 2 Width',
                  'Keypoint',
                  'RMSE - Validation Set', 'Median Epoch Training Time']  
#
columns = sorted(agg_df.columns)
discrete = [x for x in columns if agg_df[x].dtype == object]
continuous = [x for x in columns if x not in discrete]
#%%

desc = Div(text=open(os.path.join(os.path.dirname(__file__), "description.html")).read(),
           width=1000, height=100,sizing_mode="scale_both")

# Use a selection tool to pick how you want to look at the data
x_axis = Select(title="X Axis", options=sorted(agg_df.columns), value="Learning Rate")

# Use a selection tool to pick how you want to look at the data
y_axis = Select(title="Y Axis", options=sorted(agg_df.columns), value="RMSE - Validation Set")

# Specify the Architecture Results
arch = Select(title="Net Architecture", value="All",options=list(agg_df.Architecture.unique())+['All'])

# Specify the Optimizer Results
opt = Select(title="Gradient Descent Optimizer", value="All",options=list(agg_df.Optimizer.unique())+['All'])


# Cut off for scores
min_rmse = Slider(title="Maximum Validation RMSE", start=0, end=30, value=30, step=1)

# Define data source
source = ColumnDataSource(data=dict(x=[], y=[], lrate=[], doi=[], dos=[], val_rmse=[], sfilter=[]))

tooltips = [
    ("Validation RMSE", "@val_rmse"),
    ("Learning Rate", "@lrate"),
    ("Initial Dropout", "@doi"),
    ("Dropout Step", "@dos"),
    ("Starting Filter Depth", "@sfilter")]

def create_figure(df):
    kw = dict()
    xs = df[x_axis.value].values
    ys = df[y_axis.value].values
    if x_axis.value in discrete:
        kw['x_range'] = sorted(set(xs))
    if y_axis.value in discrete:
        kw['y_range'] = sorted(set(ys))
    kw['title'] = "{} vs {}".format(x_axis.value, y_axis.value)
    
    p = figure(plot_height=650, plot_width=900, 
               tools="pan, zoom_in, zoom_out, box_zoom, crosshair, lasso_select, box_select, wheel_zoom",
               toolbar_location='right',
               tooltips=tooltips, sizing_mode="scale_both", 
               background_fill_color=pal.Blues[3][0],
               border_fill_color=pal.Blues[3][0],
               **kw)
    
    p.circle(source=source,x="x", y="y", size="sfilter",
             line_color="black", 
             fill_color = pal.Magma[3][2], fill_alpha=0.85,
             hover_color='red', hover_alpha=0.5)
        
    p.grid.visible = True
    
    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.axis.axis_label_text_font = 'Calibri'
    p.axis.axis_label_text_font_size = '18pt'
    p.axis.axis_label_text_font_style = 'bold'
    p.axis.axis_line_width = 5
    p.axis.major_label_text_font_size = "14pt"
    p.axis.major_label_text_color = "white"
    p.axis.axis_label_text_color = "white"
    p.axis.major_label_text_font = 'Calibri'
    p.axis.major_tick_in = 10
    p.axis.major_tick_out = 10
    p.title.text_color = "white"
    p.title.text_font = "times"
    p.title.text_font_style = "italic"
    
    if x_axis.value in discrete:
        p.xaxis.major_label_orientation = pd.np.pi / 4
        p.axis.major_label_text_font_size = "10pt"
    
    return p

def select_data():
    
    arch_val = arch.value
    opt_val = opt.value
    selected = agg_df[
        (agg_df['RMSE - Validation Set'] <= min_rmse.value)
    ]
    if (arch_val != "All"):
        selected = selected[selected.Architecture.str.contains(arch_val)==True]
    if (opt_val != "All"):
        selected = selected[selected.Optimizer.str.contains(opt_val)==True]
    return selected

def update():
    df = select_data()
    x_name = x_axis.value
    y_name = y_axis.value
    
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        lrate=df["Learning Rate"],
        doi=df["Initial Dropout Rate"],
        dos=df["Dropout Step"],
        val_rmse=df["RMSE - Validation Set"],
        sfilter = df['Starting Filter Depth']
    )
    
    l.children[1] = row(inputs, create_figure(df))

controls = [arch, opt, min_rmse, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

# Let's make a table of data for easier data analysis
columns = [
        TableColumn(field="val_rmse", title="Validation RMSE", formatter=NumberFormatter(format='0,0.00')),
        TableColumn(field="lrate", title="Learning Rate", formatter=NumberFormatter(format='0,0.000')),
        TableColumn(field="sfilter", title="Starting Filter Depth", formatter=NumberFormatter(format='0,0')),
        TableColumn(field="doi", title="Initial Dropout", formatter=NumberFormatter(format='0,0.00')),
        TableColumn(field="dos", title="Dropout Step", formatter=NumberFormatter(format='0,0.00')),
    ]
data_table = DataTable(source=source, columns=columns, width=550, height=280)


inputs = column([c for c in controls]+[data_table], width=600, height=750)
inputs.sizing_mode = "fixed"
l = layout([
        [desc,],
    [inputs, create_figure(agg_df)]
], sizing_mode="fixed")

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "FacialKeypointDetection"