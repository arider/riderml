from ...mixture_model.dpmm import dpmm, gaussian
from ..utils import get_hex_colors
import numpy
from sklearn.datasets import load_iris
from bokeh.plotting import figure, output_file, show
from bokeh.browserlib import view
from bokeh.embed import file_html
from bokeh.models.glyphs import Circle
from bokeh.models import (
    BasicTicker, ColumnDataSource, Grid, GridPlot, LinearAxis,
    DataRange1d, PanTool, Plot, WheelZoomTool
)
from bokeh.resources import INLINE


data = load_iris().data

tmp_data = numpy.zeros([len(data), 2])
tmp_data[:, 0] = data[:, 0]
tmp_data[:, 1] = data[:, 2]

model = dpmm(tmp_data, .1, 0.000001, 5, base_distribution=gaussian)
model.fit(50)

labels = model.c
hex_colors = get_hex_colors(len(set(labels)))

f = figure(title="DPMM example", x_axis_label='x', y_axis_label='y')
output_file("DPMM_example.html")
for color_index in set(labels):
    data_inds = numpy.where(labels == color_index)[0]
    print data_inds, len(data_inds)
    f.circle(data[data_inds, 1], data[data_inds, 0], color=hex_colors[color_index])

show(f)
