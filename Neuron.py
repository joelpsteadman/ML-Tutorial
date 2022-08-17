import random

class Neuron:
    def __init__(self, num_inputs, learning_rate):
        self.learning_rate = learning_rate
        self.bias = 1

        # Set random weights
        self.weights = [];
        for i in range(num_inputs):
            self.weights.append(random.random() * 2 - 1)
        self.weights.append(self.bias)

    def activate(self, inputs):
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        if sum > 0:
            return 1
        else:
            return 0

    def train(self, inputs, desired):
        inputs.append(self.bias)
        guess = self.activate(inputs)
        error = desired - guess
        if error != 0:
            for i in range(len(inputs)):
                adjustment = self.learning_rate * error * inputs[i]
                self.weights[i] += adjustment
            self.bais = self.weights[-1]

