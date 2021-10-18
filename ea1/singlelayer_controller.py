import sys

sys.path.insert(0, "evoman")
from controller import Controller
import numpy as np
import warnings

# no. of nodes in the output layer
OUTPUT_NODES = 5


def sigmoid_activation(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1 / (1 + np.exp(-x))


class PlayerController(Controller):
    def __init__(self, hidden_nodes, output_nodes=OUTPUT_NODES):
        """
        Initializes the controller for evoman.

        :param hidden_nodes: Number of neurons in the hidden layer
        """
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

    # What is actually called controller in the demo are just weights
    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

        if self.hidden_nodes > 0:
            # Preparing the weights and biases from the controller of the hidden layer
            # The encoding is [biases_hidden_layer, weights_hidden_layer, biases_output_layer, weights_output_layer]

            # Biases for the n hidden neurons
            biases1 = controller[:self.hidden_nodes].reshape(1, self.hidden_nodes)

            # Weights for the connections from the inputs to the hidden nodes
            weights1_slice = len(inputs) * self.hidden_nodes + self.hidden_nodes
            weights1 = controller[self.hidden_nodes:weights1_slice].reshape((len(inputs), self.hidden_nodes))

            # Outputs activation first layer.
            output1 = sigmoid_activation(inputs.dot(weights1) + biases1)

            # Preparing the weights and biases from the controller of layer 2
            bias2 = controller[weights1_slice:weights1_slice + self.output_nodes].reshape(1, self.output_nodes)
            weights2 = controller[weights1_slice + self.output_nodes:].reshape((self.hidden_nodes, self.output_nodes))

            # Outputting activated second layer. Each entry in the output is an action
            output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
        else:
            bias = controller[:self.output_nodes].reshape(1, self.output_nodes)
            weights = controller[self.output_nodes:].reshape((len(inputs), self.output_nodes))

            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # takes decisions about sprite actions
        actions = [1 if score > 0.5 else 0 for score in output]

        # [left, right, jump, shoot, release]
        return actions
