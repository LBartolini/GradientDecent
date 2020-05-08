import numpy as np
from gd import *

def fun(x):
	x= np.array(x)
	return x**2 + np.sin(10*x)

results_x, results_y = [], []
for i in range(10):
	minim_x, minim_y, step = gradientDescent(data_to_display=np.arange(-5, 5, 0.001), funct=fun, lr=1e-1,
											 display=False, precision_points=5, logspace_limits=(-2, 1), steps_to_display=1,
											 explorer_reduction=2.5, base_offset=0.4,
											 explorer_importance=5, treshold_to_stop=1e1000)
	results_x.append(minim_x)
	results_y.append(minim_y)
	print(f"The minimum of the function is x= {minim_x} at y= {minim_y}, reached with {step} step. Epoch {i}")

print(f"The Real Global Minimum is x= {results_x[np.argmin(results_y)]} at y= {np.min(results_y)}")