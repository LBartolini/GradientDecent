import numpy as np
from gd import *

def fun(x):
	x= np.array(x)
	return x**2

minim, step = gradientDecent(data_to_display=np.arange(-5, 5, 0.1), funct=fun, lr=1e-2, display=True, precision_points=5, logspace_limits=(-2, 2), steps_to_display=20)
print(f"The minimum of the function is {minim}, reached with {step} step.")