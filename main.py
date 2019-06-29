import numpy as np
from gd import *

def fun(x):
	x= np.array(x)
	#return x**2 + np.sin(5*x)
	return x**2 - 2*x

results = []
for i in range(10):
	minim, step = gradientDescent(data_to_display=np.arange(-5, 5, 0.001), funct=fun, lr=1e-2, display=False, precision_points=10, logspace_limits=(-2, 1), steps_to_display=10, explorer_reduction=0.4)
	results.append(minim)
	print(f"The minimum of the function is {minim}, reached with {step} step. Epoch {i}")

print(results)