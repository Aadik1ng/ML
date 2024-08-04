import numpy as np
from ipywidgets import interact
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.io import push_notebook

output_notebook()

def local_regression(x0, X, Y, tau):
    X_aug = np.c_[np.ones(len(X)), X]
    weights = np.exp(-np.sum((X_aug - np.r_[1, x0])**2, axis=1) / (2 * tau**2))
    W = np.diag(weights)
    beta = np.linalg.pinv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ Y
    return np.r_[1, x0] @ beta

n = 1000
X = np.linspace(-3, 3, num=n)
Y = np.log(np.abs(X**2 - 1) + 0.5) + np.random.normal(scale=0.1, size=n)

def plot_lwr(tau):
    domain = np.linspace(-3, 3, num=300)
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    p = figure(width=400, height=400, title=f'tau={tau}')
    p.scatter(X, Y, alpha=0.3)
    p.line(domain, prediction, line_width=2, color='red')
    return p

show(gridplot([[plot_lwr(10), plot_lwr(1)], [plot_lwr(0.1), plot_lwr(0.01)]]))

domain = np.linspace(-3, 3, num=100)
plot = figure(title='Locally Weighted Regression', x_axis_label='X', y_axis_label='Y')
plot.scatter(X, Y, alpha=0.3)
model = plot.line(domain, [local_regression(x0, X, Y, 1) for x0 in domain], line_width=2, color='red')
handle = show(plot, notebook_handle=True)

def interactive_update(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    model.data_source.data['y'] = prediction
    push_notebook(handle=handle)

interact(interactive_update, tau=(0.01, 3.0, 0.01))
