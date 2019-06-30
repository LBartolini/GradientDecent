import matplotlib.pyplot as plt
import numpy as np
import random as r


def gradientDescent(funct, data_to_display=np.arange(-5, 5, 0.001), precision_points=3, lr=1e-2, display=False, steps_to_display=300, logspace_limits=(-1, 0), explorer_reduction=10, precision_tangent=1e-4, treshold_to_stop=1e3, base_offset=0.5, explorer_importance=2):
	minimum_x = None
	minimum_y = 1e1000
	if precision_points < 3:
		precision_points = 3
	points = np.array([r.choice(data_to_display) for _ in range(precision_points)])
	curve = funct(data_to_display)
	if display:
		plt.plot(data_to_display, curve, c='green')
		plt.scatter(points, funct(points), color='red')
		plt.show()
	yp_points = [5 for _ in range(precision_points)]
	step = 0
	min_value = 1

	while np.abs(min_value) > precision_tangent and step < treshold_to_stop:
		step += 1
		yp_points = deriv(funct, points)
		for i in points:
			if funct(i) < minimum_y:
				#minimum_x = points[np.argmin(np.abs(yp_points))]
				minimum_x = i
				minimum_y = funct(minimum_x)
		min_value = yp_points[np.argmin(np.abs(yp_points))]
		if min_value>0:
			points = [minimum_x-np.abs(lr*i*np.abs(min_value)) for i in np.logspace(logspace_limits[0], logspace_limits[1], precision_points-2)]
		else:
			points = [minimum_x+np.abs(lr*i*np.abs(min_value)) for i in np.logspace(logspace_limits[0], logspace_limits[1], precision_points-2)]
		
		points.append(minimum_x+np.abs(lr*(10**logspace_limits[1]))+base_offset+(np.abs(min_value**explorer_importance)/explorer_reduction))
		points.append(minimum_x+np.abs(lr*(10**logspace_limits[1]))-base_offset-(np.abs(min_value**explorer_importance)/explorer_reduction))

		if display and step%steps_to_display==0:
			plt.plot(data_to_display, curve, c='green')
			plt.scatter(points, funct(points), color='red')
			plt.scatter(minimum_x, funct(minimum_x), color='blue')
			plt.show()

	if display:	
		plt.plot(data_to_display, curve, c='green')
		plt.scatter(points, funct(points), color='red')
		plt.show()
	minimum_y = funct(minimum_x)
	return minimum_x, minimum_y, step

def deriv(funct, x):
	h = 1e-5
	x = np.array(x)
	return (funct(x+h)-funct(x-h))/(2*h)