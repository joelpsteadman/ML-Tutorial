import matplotlib.pyplot as plt
import numpy as np
import random

x_max = 100
y_max = 100
x_min = 0
y_min = 0

y_range = y_max - y_min

slope = (random.random() * 0.5) - 0.25 # between -0.25 and 0.25
print("Slope: ", slope)

intercept = random.random() * (y_range / 2) + 25 # between 25 and 75
print("Intercept: ", intercept)

def fun(x):
    return slope * x + intercept

x = np.linspace(x_min, x_max, 100)
y = fun(x)
plt.plot(x, y, '-r', label='y=2x+1')


num_points = 500
x_points = []
y_points = []
for i in range(num_points):
  x_points.append(random.random() * x_max)
  y_points.append(random.random() * y_max)

# x_points = np.array(x_points)
# y_points = np.array(y_points)

# for i in range(num_points):
plt.plot(x_points, y_points, 'o')
plt.show()
