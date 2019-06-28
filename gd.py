import matplotlib.pyplot as plt
import numpy as np
import random as r


def gradientDecent(funct, data_to_display=np.arange(-5, 5, 0.3), precision_points=3, lr=1e-2, display=False, steps_to_display=300, logspace_limits=(-1, 0), explorer_reduction=10, precision_tangent=1e-4):
	minimum = None

	points = np.array([r.choice(data_to_display) for _ in range(precision_points)])
	curve = funct(data_to_display)
	if display:
		plt.plot(data_to_display, curve, c='green')
		plt.scatter(points, funct(points), color='red')
		plt.show()
	yp_points = [5 for _ in range(precision_points)]
	step = 0
	min_value = 1

	while np.abs(min_value) > precision_tangent:
		step += 1
		yp_points = deriv(funct, points)
		minimum = points[np.argmin(np.abs(yp_points))]
		min_value = yp_points[np.argmin(np.abs(yp_points))]
		diff = min_value*lr*np.abs(min_value) #da cancellare
		if min_value>0:
			points = [minimum-np.abs(lr*i*np.abs(min_value)) for i in np.logspace(logspace_limits[0], logspace_limits[1], precision_points-1)]
			points.append(minimum+np.abs(lr*np.abs(min_value)*(10**logspace_limits[1])/explorer_reduction))
		else:
			points = [minimum-np.abs(lr*i*np.abs(min_value)) for i in np.logspace(logspace_limits[0], logspace_limits[1], precision_points-1)]
			points.append(minimum+np.abs(lr*np.abs(min_value)*(10**logspace_limits[1])/explorer_reduction))

		if display and step%steps_to_display==0:
			plt.plot(data_to_display, curve, c='green')
			plt.scatter(points, funct(points), color='red')
			plt.show()
			print(min_value, np.abs(diff)) #da cancellare

	if display:	
		plt.plot(data_to_display, curve, c='green')
		plt.scatter(points, funct(points), color='red')
		plt.show()

	return minimum, step

def deriv(funct, x):
	h = 1e-5
	x = np.array(x)
	return (funct(x+h)-funct(x))/h