import random

# TODO customizable activation functions

class Neuron:
    def __init__(self, num_inputs, activation_function, learning_rate):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.bias = random.uniform(-100, 100) #TODO don't hardcode

        # Sets random weights between -1 and 1
        self.weights = []
        for i in range(num_inputs):
            self.weights.append(random.uniform(-1, 1))

    def activate(self, inputs):
        sum = 0
        for i in range(len(inputs)):
            sum += inputs[i] * self.weights[i]
        sum += self.bias
        if self.activation_function == "Binar Step":
            if sum > 0:
                return 1
            else:
                return 0
        if self.activation_function == "ReLU":
            if sum > 0:
                return sum
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

    def backpropagate(self, inputs, desired):
        guess = self.activate(inputs)
        error = desired - guess
        if error > 0:
            for i in range(len(inputs)):
                adjustment = self.learning_rate * error * inputs[i]
                self.weights[i] += adjustment
            bias_adjustment = self.learning_rate * error
            self.bias += bias_adjustment
        desires = []
        if desired >= self.activate(inputs):
            for weight in self.weights:
                desires.append(self.activate([weight]))
                # if weight > 0:
                #     desires.append(1)
                # else:
                #     desires.append(0)
        else:
            for weight in self.weights:
                desires.append(self.activate([weight]))
                # if weight > 0:
                #     desires.append(0)
                # else:
                #     desires.append(1)
        return desires

    def show(self, num = ""):
        if num == "":
            prefix = ""
        else:
            prefix = "\t\t"
            print("\t\t", "Neuron #", num, ":", sep='')
        rounded_weights = [round(weight, 3) for weight in self.weights]
        print(prefix, "Weights: ", rounded_weights, sep='')
        print(prefix, "Bias:    ", round(self.bias, 3), sep='')

    # Currently only works for linear functions (2 inputs)
    def function(self):
        params = {}
        # x_weight = network.weights[0]
        # y_weight = network.weights[1]
        # m = round(-x_weight/y_weight, 2)
        # b = round(-network.bias/y_weight, 2)

        params['m'] = round(-self.weights[0]/self.weights[1], 2)
        params['b'] = round(-self.bias/self.weights[1], 2)
        return params