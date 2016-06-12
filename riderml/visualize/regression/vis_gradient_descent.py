import numpy
from ...regression.gradient_descent import gradient_descent
from ...util.loss_functions import linear, dlinear

from bokeh.models import Plot
from bokeh.plotting import figure, output_file, show

from bokeh.embed import file_html
from bokeh.models.glyphs import Circle, Line
from bokeh.models import (
    ColumnDataSource, Grid, GridPlot, LinearAxis, Plot, Range1d)
from bokeh.resources import INLINE, CDN
from bokeh.browserlib import view
from bokeh.plotting import figure, show, output_file


if __name__ == "__main__":
    """
    plot multiple solutions as the number of iterations increases
    """
    n = 500
    x = numpy.ones([n, 2])
    x[:, 1] = range(n)
    y = numpy.array(range(n)) / 4 * numpy.sin(numpy.pi * numpy.linspace(0, 1, n) * 4)
    y = y.reshape(y.shape[0], 1)
    inds = numpy.random.permutation(range(n))
    cv = int(.8 * n)
    train_x = x[inds[:cv], :]
    test_x = x[inds[cv:], :]
    train_y = y[inds[:cv], :]
    test_y = y[inds[cv:], :]

    n_iterations = 50
    p_its = 5
    predictions = []
    prev_coef = None
    for i in range(p_its):
        coef = gradient_descent(
            linear,
            dlinear,
            train_x,
            train_y,
            theta=prev_coef,
            iterations=n_iterations)
        predicted = numpy.dot(test_x, coef)
        predictions.append(predicted)

        prev_coef = coef

    HEX_RAINBOW = [
        '#f80c12',
        '#feae2d',
        '#69d025',
        '#4444dd',
        '#3b0cbd',
    ]
    HEX_RAINBOW.reverse()



    f = figure(title="", x_axis_label='x', y_axis_label='y', x_range=[0, 500])
    output_file("gradient_descent.html")
    f.circle(train_x[:, 1], train_y[:, 0], legend='training data', color='#ff0000', alpha=.5)
    f.circle(test_x[:, 1], test_y[:, 0], legend='test data', color='#3300ff')

    for i in range(p_its):
        f.line(test_x[:, 1], predictions[i][:, 0],
               legend='iteration {}'.format(i * n_iterations),
               color=HEX_RAINBOW[i],
               line_width=3)
    f.legend.orientation = "top_left"
    show(f)

