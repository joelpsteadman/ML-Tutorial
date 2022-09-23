from Neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function, learning_rate):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.neurons = []
        for i in range(num_neurons):
            neuron = Neuron(num_inputs, activation_function, learning_rate)
            self.neurons.append(neuron)

    def show(self, num = ""):
        if num == "":
            prefix = ""
        else:
            prefix = "\t"
            print(prefix, "Layer #", num, ":", sep='')
        print(prefix, "Depth: ", self.num_neurons, sep='')
        iterator = 1
        for neuron in self.neurons:
            neuron.show(iterator)
            iterator += 1