import sys # for taking arguments from the command line
from Utilities import Logger
from Network import Network
# from Layer import Layer
# from Neuron import Neuron
from Graph import Graph
from Data import Data_Set
import numpy as np

# Set up logging
command_line_arguments = sys.argv
debug = False
if "-v" in command_line_arguments or "--verbose" in command_line_arguments:
    debug = True
logger = Logger(debug)

# Global variable
learning_rate = 0.0001
if "-l" in command_line_arguments:
    learning_rate = command_line_arguments[command_line_arguments.index("-l") + 1]
elif "--learning_rate" in command_line_arguments:
    learning_rate = command_line_arguments[command_line_arguments.index("--learning_rate") + 1]

logger.debug("learning_rate =", learning_rate)
logger.debug("Arguments:", command_line_arguments)

num_training = 500000
num_testing = 10000
num_iterations = 1

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

# create a network with 2 inputs, 3 hidden layers of 5 networks, and 1 output
'''
    n1  n1  n1
x   n2  n2  n2
    n3  n3  n3  ouput
y   n4  n4  n4
    n5  n5  n5
'''
network = Network(2, [10], learning_rate)
# network.show()

# TODO does increasing the num_iterations do the same thing as increasing the learning rate?
# Train data
for j in range(num_iterations):
    for i in range(num_training):
        logger.display_progress("Training: ", i, num_training)
        network.train([x_points[i], y_points[i]], data.desired(x_points[i], y_points[i]))
logger.display_progress("Training: ", num_iterations, num_iterations, final=True)

# Assess results
training_data = data.generate_testing_set(num_testing)
num_right = 0
for x, y in zip(training_data[0], training_data[1]):
    desired = data.desired(x, y)
    guess = network.activate([x, y])[0]
    # logger.debug("Guessed:", guess, "but should have been", desired)
    if data.desired(x, y) == network.activate([x, y])[0]:
        num_right += 1
percent_correct = round((num_right / num_testing) * 100, 2)


# Display answers
# x_positives = []
# y_positives = []
# x_negatives = []
# y_negatives = []
# iterator = 0
# for x, y in zip(training_data[0], training_data[1]):
#     logger.display_progress("Generating graph: ", iterator, num_testing)
#     answer = data.desired(x, y)
#     if answer == 1:
#         x_positives.append(x)
#         y_positives.append(y)
#     else:
#         x_negatives.append(x)
#         y_negatives.append(y)
#     iterator += 1
# graph.show(x_positives, y_positives, x_negatives, y_negatives)

# x_weight = network.weights[0]
# y_weight = network.weights[1]
# m = round(-x_weight/y_weight, 2)
# b = round(-network.bias/y_weight, 2)
# logger.debug("Weights: ", network.weights)
# logger.debug("Bias: ", network.bias)
# logger.debug("m = ", m)
# logger.debug("b = ", b)
# logger.debug("y = ", m, "x", " + ", b)

logger.info("Neuron is ", percent_correct, "% correct", delimiter='')

# logger.debug("Expected", round(data.slope, 2), "x +", data.intercept)
# logger.debug("But got ", m, "x +", b)

x_positives = []
y_positives = []
x_negatives = []
y_negatives = []

# Display results
iterator = 0
for x, y in zip(training_data[0], training_data[1]):
    logger.display_progress("Generating graph: ", iterator, num_testing)
    guess = network.activate([x, y])[0]
    if guess == 1:
        x_positives.append(x)
        y_positives.append(y)
    else:
        x_negatives.append(x)
        y_negatives.append(y)
    iterator += 1

logger.debug(len(x_positives), "positive values")
logger.debug(len(x_negatives), "negative values")
network.show()

# debugging by mapping a lot of neuron slopes
# def neuron_separation_function(x, m, b):
#     return m*x + b

# x = np.linspace(x_min, x_max, x_max)
# for layer in network.layers:
#     for neuron in layer.neurons:
#         values = neuron.function()
#         y = neuron_separation_function(x, abs(values['m']), values['b'])
#         graph.create_reference_line(x, y)

graph.show(x_positives, y_positives, x_negatives, y_negatives)