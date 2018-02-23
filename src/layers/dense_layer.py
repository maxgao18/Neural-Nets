import numpy as np
from functions import LeakyRELU
from functions import RELU
from layer import Layer

class DenseLayer(Layer):
    # Args:
    #   layer_shape - a 2-tuple of ints (number of neurons on current layer, number of neurons on previous layer)
    #   weights (optional) - a 2D np array of the weights
    #   biases (optional) a 1D np array of the biases
    def __init__(self, input_shape, output_shape, weights=None, biases=None, activation_function=RELU):
        super(DenseLayer,self).__init__(input_shape=input_shape,
                                        output_shape=output_shape,
                                        activation_function=activation_function)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randn(output_shape, input_shape)

        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.random.randn(output_shape)

    # Similar to feed forward but without squashing
    def getactivations(self, inputs):
        return np.dot(self.weights, inputs) + self.biases

    # Feeds the input through the layer and uses leaky relu as an logistic function
    # Args:
    #   input_activations - a 1D np array of the previous activations
    def feedforward(self, inputs):
        return self.activation_function.func(self.getactivations(inputs))

    # Returns the gradients for the weights, biases, and the deltas for the previous layer
    def backprop (self, prev_fz_activations, d_prev_z_activations, curr_deltas):
        biasDeltas = curr_deltas

        prevDeltas = self.getdeltas(d_prev_z_activations, curr_deltas)
        weightDeltas = np.dot(np.array([curr_deltas]).transpose(), np.array([prev_fz_activations]))

        return weightDeltas, biasDeltas, prevDeltas

    def getdeltas(self, d_prev_z_activations, curr_deltas):
        prevDeltas = np.dot(self.weights.transpose(), curr_deltas) * d_prev_z_activations
        return prevDeltas

    # Updates layers parameters
    # Args:
    #   d_weights - 2D np array determining how much to change the weights by
    #   d_biases - 1D np array determining how much to change the biases by
    def update(self, d_weights, d_biases):
        self.weights += d_weights
        self.biases += d_biases

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases