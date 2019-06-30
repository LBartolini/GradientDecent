During this project I have experienced what really means Gradient Descent.

I'm sure I didn't do it as you usually see but it works so I'm happy with it.

The real "library" is called gd.py which has a function called gradientDescent.

Parameters:
 * funct(necessary):
Receive in input a function that is used to compute the values to be displayed and to compute the derivative.
 * data_to_display:
Receive an array of numbers that can be seen as the X axis, it is usually given a np.arange
 * precision_points:
Receive an integer value that tell the function how many points to use to descend the gradient.
 * lr:
Receive a float value that can be seen as the learning rate. It is used to reduce the step towards the minimum.
 * display:
Receive a bool wich tells if to display the graph using matplotlibs.
 * steps_to_display:
Receive an integer value that is used to display every n steps. In any case if you want to display the graphs remember to put display at True.
 * logspace_limits:
Receive an array of Two elements that are the limits of the method np.logspace(a, b, lenght), so a and b are taken from this particular parameter.
 * explorer_reduction:
Receive an integer value wich tells how large the steps of the explorer have to be. The "explorer" is one of the points wich goes far from the brothers and with the opposite direction in order to find a better minimum. 
P.S. This part has to be done.
 * precision_tangent:
Receive a float value that tells when to stop to search. It refers to the derivative of the best of the points used.
 * treshold_to_stop:
Receive an integer value that is used to stop the precess if it's taking too many steps to converge.
 * base_offset:
Receive a float value, tells the minimum distance between the explorers and the main point.
 * explorer_importance:
Receive an integer value, tells the importance of the centre to the explorer. Bigger values say to explorer to search far away from their friends.


The main.py file is just an example to see how it is supposed to work.

Enjoy this amazing project!