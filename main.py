import Neuron
import matplotlib.pyplot as plt
import numpy as np
import random
from Utilities import Logger

logger = Logger(debug=True)

learning_rate = 0.00001
num_training = 5000
num_testing = 10000
num_iterations = 1000

x_max = 100
y_max = 100
x_min = 0
y_min = 0

y_range = y_max - y_min

slope = (random.random() * 2) - 1 # between -1 and 1
logger.debug("Slope: ", slope)

intercept = random.random() * (y_range / 2)
if slope < 0:
    intercept += 50
logger.debug("Intercept: ", intercept)

def fun(x, intercept):
    return slope * x + intercept

x = np.linspace(x_min, x_max, x_max)
y = fun(x, intercept)
# TODO don't plot outside of the x/y min/max
plt.plot(x, y, color='black')


x_points = []
y_points = []
for i in range(num_training):
  x_points.append(random.random() * x_max)
  y_points.append(random.random() * y_max)

def desired(x, y):
  if y > fun(x, intercept):
    return 1
  else:
    return 0

neuron = Neuron.Neuron(2, learning_rate)

for j in range(num_iterations):
  logger.display_progress("Training: ", j, num_iterations)
  for i in range(num_training):
    neuron.train([x_points[i], y_points[i]], desired(x_points[i], y_points[i]))

num_right = 0
x_test_cases = []
y_test_cases = []
for i in range(num_testing):
  x = random.random() * x_max
  y = random.random() * y_max
  x_test_cases.append(x)
  y_test_cases.append(y)
  if desired(x, y) == neuron.activate([x, y]):
      num_right += 1
x_weight = neuron.weights[0]
y_weight = neuron.weights[1]
m = round(-x_weight/y_weight, 2)
b = round(-neuron.bias/y_weight, 2)
logger.debug("Weights: ", neuron.weights)
logger.debug("Bias: ", neuron.bias)
logger.debug("m = ", m)
logger.debug("b = ", b)
logger.debug("y = ", m, "x", " + ", b)

logger.log("Neuron is ", round((num_right / num_testing) * 100, 2), "% correct", delimiter='')

logger.debug("Expected", round(slope, 2), "x +", intercept)
logger.debug("But got ", m, "x +", b)

x_positives = []
y_positives = []
x_negatives = []
y_negatives = []

# Display results
for i in range(num_testing):
  logger.display_progress("Generating graph: ", i, num_testing)
  x = x_test_cases[i]
  y = y_test_cases[i]
  guess = neuron.activate([x, y])
  if guess == 1:
    x_positives.append(x)
    y_positives.append(y)
  else:
    x_negatives.append(x)
    y_negatives.append(y)

x = np.array(x_positives)
y = np.array(y_positives)
logger.debug(len(x), " positive values")
plt.scatter(x, y, color='green')

x = np.array(x_negatives)
y = np.array(y_negatives)
logger.debug(len(x), " negative values")
plt.scatter(x, y, color='red')

plt.show()