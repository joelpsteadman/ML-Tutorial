from Layer import Layer

# Parameters: 
'''
Number of layers
Number of neurons in each layer
Activation function
Learning rate
...?
'''
class Network:
    # hidden_layers is a list of neuron counts
    def __init__(self, num_inputs, hidden_layers, learning_rate):
        # TODO create separate variables for the input and output layers
        self.layers = []
        previous_layer_outputs = num_inputs
        self.learning_rate = learning_rate
        for neuron_count in hidden_layers:

            layer = Layer(num_neurons = neuron_count, 
                          num_inputs = previous_layer_outputs, 
                        activation_function = "ReLU", 
                          learning_rate = self.learning_rate)

            previous_layer_outputs = neuron_count
            self.layers.append(layer)
        
        output = Layer(num_neurons = 1, 
                        num_inputs = previous_layer_outputs, 
                        activation_function = "Binar Step", 
                        learning_rate = self.learning_rate)
        self.layers.append(output)

    # Returns the output of the final layer as a list of values
    def activate(self, inputs):
        # activate all neurons in each layer
        for layer in self.layers:
            outputs = []
            for neuron in layer.neurons:
                outputs.append(neuron.activate(inputs))
            inputs = outputs # outputs of previous layer become inputs for next layer
        final_output = inputs
        return final_output

    # TODO this function doesn't work when hidden layers have different depths
    def train(self, inputs, desired):
        # train all neurons in each layer
        # for layer in self.layers:
        #     outputs = []
        #     for neuron in layer.neurons:
        #         outputs.append(neuron.activate(inputs))
        #         neuron.train(inputs, desired)
        #     inputs = outputs # outputs of previous layer become inputs for next layer
        activations = []
        for layer in self.layers:
            outputs = []
            for neuron in layer.neurons:
                outputs.append(neuron.activate(inputs))
                neuron.train(inputs, desired)
            activations.append(outputs)
            inputs = outputs # outputs of previous layer become inputs for next layer

        index = -1
        
        desired_inputs = []
        desired_outputs = [desired]
        for _ in range(len(self.layers)): # for each layer (in reverse order)
            for j in range(len(self.layers[index].neurons)): # for each neuron 
                desired_inputs.append(self.layers[index].neurons[j].backpropagate(activations[len(activations) + index - 1], desired_outputs[j]))
            sums = [0] * len(desired_inputs[0])
            for set in desired_inputs:
                for k in range(len(set)):
                    sums[k] += set[k]
            for sum in sums:
                desired_outputs.append(sum/len(desired_inputs))
            desired_inputs = desired_inputs.copy() # TODO this can't be right, can it?
            index -= 1


        # reversed_layers = reversed(self.layers)

        # for layer in reversed_layers:
        #     outputs = []
        #     for neuron in layer.neurons:
        #         outputs.append(neuron.activate(inputs))
        #         neuron.train(inputs, desired)
        #     inputs = outputs # outputs of previous layer become inputs for next layer

    def show(self):
        print("Learning Rate: ", self.learning_rate, sep='')
        iterator = 1
        for layer in self.layers:
            layer.show(iterator)
            iterator += 1