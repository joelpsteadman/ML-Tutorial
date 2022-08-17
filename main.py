from audioop import bias
import Neuron
import matplotlib.pyplot as plt
import numpy as np
import random
import time

start = time.time()

learning_rate = 0.00001
num_tests = 1000

x_max = 100
y_max = 100
x_min = 0
y_min = 0

y_range = y_max - y_min

slope = (random.random() * 2) - 1 # between -1 and 1
print("Slope: ", slope)

intercept = random.random() * (y_range / 2)
if slope < 0:
    intercept += 50
print("Intercept: ", intercept)

def fun(x, intercept):
    return slope * x + intercept

x = np.linspace(x_min, x_max, x_max)
y = fun(x, intercept)
plt.plot(x, y)


num_points = 500
x_points = []
y_points = []
for i in range(num_points):
  x_points.append(random.random() * x_max)
  y_points.append(random.random() * y_max)

# x_points = np.array(x_points)
# y_points = np.array(y_points)

# for i in range(num_points):
# plt.plot(x_points, y_points, 'o')

def desired(x, y):
  if y > fun(x, intercept):
    return 1
  else:
    return 0

neuron = Neuron.Neuron(2, learning_rate)

for j in range(10000):
  for i in range(num_points):
    neuron.train([x_points[i], y_points[i]], desired(x_points[i], y_points[i]))

num_right = 0
x_test_cases = []
y_test_cases = []
for i in range(num_tests):
  x = random.random() * x_max
  y = random.random() * y_max
  x_test_cases.append(x)
  y_test_cases.append(y)
  if desired(x, y) == neuron.activate([x, y, neuron.bias]):
      num_right += 1

print("Weights: ", neuron.weights)

print("Neuron is ", round((num_right / num_tests) * 100, 2), "% correct")

end = time.time()

time_to_complete = round(end - start, 2)
print("Completed in ", time_to_complete, " seconds")

x_positives = []
y_positives = []
x_negatives = []
y_negatives = []

# Display results
for i in range(num_tests):
  x = x_test_cases[i]
  y = y_test_cases[i]
  guess = neuron.activate([x, y, neuron.bias])
  if guess == 1:
    x_positives.append(x)
    y_positives.append(y)
  else:
    x_negatives.append(x)
    y_negatives.append(y)

#day one, the age and speed of 13 cars:
x = np.array(x_positives)
y = np.array(y_positives)
print(len(x), " positive values")
plt.scatter(x, y)

x = np.array(x_negatives)
y = np.array(y_negatives)
print(len(x), " negative values")
plt.scatter(x, y)

plt.show()