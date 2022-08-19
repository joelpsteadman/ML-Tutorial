import random

class Neuron:
    def __init__(self, num_inputs, learning_rate):
        self.learning_rate = learning_rate
        self.bias = 1

        # Set random weights
        self.weights = []
        for i in range(num_inputs):
            self.weights.append(random.random() * 2 - 1)

    def activate(self, inputs):
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        sum += self.bias
        if sum > 0:
            return 1
        else:
            return 0

    def train(self, inputs, desired):
        guess = self.activate(inputs)
        error = desired - guess
        if error != 0:
            for i in range(len(inputs)):
                adjustment = self.learning_rate * error * inputs[i]
                self.weights[i] += adjustment
            bias_adjustment = self.learning_rate * error
            self.bias += bias_adjustment
