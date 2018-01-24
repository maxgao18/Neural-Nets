from functions import LeakyRELU

from copy import deepcopy
import numpy as np

# leaky relu function
def func (z):
    return LeakyRELU.func(z)

def func_deriv(z):
    return LeakyRELU.func_deriv(z)

class RecurrentLayer:
    def __init__(self, layer_shape, weights=None, biases=None, past_weights=None):
        self.layer_shape = layer_shape
        self.output_shape = layer_shape[0]
        self.past_state = np.zeros(layer_shape[0])

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randn(layer_shape[0], layer_shape[1])/np.sqrt(layer_shape[0]*layer_shape[1])

        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.random.randn(layer_shape[0])

        if past_weights is not None:
            self.past_weights = past_weights
        else:
            self.past_weights = np.random.randn(layer_shape[0], layer_shape[0])/(layer_shape[0]+0.00)

    # Feed forward without squashing and saving
    def get_activations(self, input_activations):
        curr_activ = np.dot(self.weights, input_activations) + np.dot(self.past_weights, self.past_state) + self.biases
        return deepcopy(self.past_state), curr_activ

    # Feeds the input through the layer and uses leaky relu as an logistic function
    # Args:
    #   input_activations - a 1D np array of the previous activations
    def feed_forward(self, input_activations):
        ps, cs = self.get_activations(input_activations)
        self.past_state = cs
        return func(deepcopy(self.past_state))

    # Returns the gradients for the weights, biases, and the deltas for the previous layer
    def backprop(self, prev_z_activ, z_activations, deltas):
        prevDeltas = np.dot(self.weights.transpose(), deltas) * func_deriv(z_activations)
        biasDeltas = deltas
        weightDeltas = np.dot(np.array([deltas]).transpose(), np.array([func(z_activations)]))
        pastWeightDeltas = np.dot(np.array([deltas]).transpose(), np.array([prev_z_activ]))

        return weightDeltas, pastWeightDeltas, biasDeltas, prevDeltas

    # Updates layers parameters
    # Args:
    #   d_weights - 2D np array determining how much to change the weights by
    #   d_biases - 1D np array determining how much to change the biases by
    def update(self, d_weights, d_past_weights, d_biases):
        self.weights += d_weights
        self.past_weights += d_past_weights
        self.biases += d_biases

    def forget_past(self):
        self.past_state = np.zeros(self.layer_shape[0])

    def get_output_shape(self):
        return self.output_shape