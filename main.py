import Neuron
import numpy as np
from Data import Data_Set
from Graph import Graph
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

data = Data_Set(x_min, x_max, y_min, y_max)
data.generate_delimiter()
graph = Graph(num_training, num_testing, x_max, y_max, x_min, y_min)

logger.debug("Slope:", data.slope)
logger.debug("Intercept:", data.intercept)

x = np.linspace(x_min, x_max, x_max)
y = data.separation_function(x)
graph.create_reference_line(x, y)

training_set = data.generate_training_set(num_training)
x_points = training_set[0]
y_points = training_set[1]

neuron = Neuron.Neuron(2, "Binar Step", learning_rate)

# TODO does increasing the num_iterations do the same thing as increasing the learning rate?
for j in range(num_iterations):
  logger.display_progress("Training: ", j, num_iterations)
  for i in range(num_training):
    neuron.train([x_points[i], y_points[i]], data.desired(x_points[i], y_points[i]))

# Train data
training_data = data.generate_testing_set(num_testing)
num_right = 0
for x, y in zip(training_data[0], training_data[1]):
    if data.desired(x, y) == neuron.activate([x, y]):
        num_right += 1
percent_correct = round((num_right / num_testing) * 100, 2)

x_weight = neuron.weights[0]
y_weight = neuron.weights[1]
m = round(-x_weight/y_weight, 2)
b = round(-neuron.bias/y_weight, 2)
logger.debug("Weights: ", neuron.weights)
logger.debug("Bias: ", neuron.bias)
logger.debug("m = ", m)
logger.debug("b = ", b)
logger.debug("y = ", m, "x", " + ", b)

logger.info("Neuron is ", percent_correct, "% correct", delimiter='')

logger.debug("Expected", round(data.slope, 2), "x +", data.intercept)
logger.debug("But got ", m, "x +", b)

x_positives = []
y_positives = []
x_negatives = []
y_negatives = []

# Display results
for x, y in zip(training_data[0], training_data[1]):
  logger.display_progress("Generating graph: ", i, num_testing)
  guess = neuron.activate([x, y])
  if guess == 1:
    x_positives.append(x)
    y_positives.append(y)
  else:
    x_negatives.append(x)
    y_negatives.append(y)

logger.debug(len(x_positives), "positive values")
logger.debug(len(x_negatives), "negative values")
graph.show(x_positives, y_positives, x_negatives, y_negatives)