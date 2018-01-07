import numpy as np
import functions as fn

class DenseLayer:# -- Class for the dense layer
    # Args:
    #   layer_shape (tuple): a 2-tuple (number of neurons on current layer, number of neurons on previous layer)
    def __init__(self, layer_shape, logistic_func="sig", weights=None, biases=None):
        self.layer_shape = layer_shape

        if logistic_func=="sig":
            self.logistic_func = fn.Sigmoid
        elif logistic_func=="relu":
            self.logistic_func = fn.ReLU
        elif logistic_func=="leakyrelu":
            self.logistic_func = fn.LeakyReLU
        elif logistic_func=="tanh":
            self.logistic_func = fn.TanH
        elif logistic_func=="softmax":
            self.logistic_func = fn.SoftMax

        # Weights is a 2D list w[x][y] where x is the neuron number in the current layer and
        # y is the neuron number on the previous layer
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randn(layer_shape[0], layer_shape[1])

        # Biases is a list biases for each neuron
        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.random.randn(layer_shape[0])

    # Calculates the activation of the layer given a list of activations
    # Args:
    #   activations (1D np array): a np array with the activations in the previous layer
    def feed_forward(self, activations):
        return self.logistic_func(np.dot(self.weights, activations) + self.biases)

    # Returns all weights in the layer (2D Array)
    def get_all_weights(self):
        return self.weights

    # Returns all biases in the layer
    def get_all_biases(self):
        return self.biases

    # Returns weights in the layer connecting to a neuron (1D Array)
    def get_weights(self, index):
        return self.weights[index]

    # Returns all biases in the layer
    def get_biases(self, index):
        return self.biases[index]

    # Returns layer shape of network
    def get_layer_shape(self):
        return self.layer_shape

    # Returns the total number of neurons
    def get_num_neurons(self):
        return self.layer_shape[0]

    # Sets the weights and biases
    # Args:
    #   weights (2D np array): a np array of weights. Size of weights expected to be (number of neurons on current
    #       layer, number of neurons on previous layer)
    #   biases (1D np array): a np array of biases. Size of biases expected to be (number of neurons on current layer)
    # Returns:
    #   the shape of the layer created upon success
    #   error message upon failure
    def set_weights_biases (self, weights, biases):
        # Check that arrays are compatable
        if not len(weights) == len(biases):
            return "Failed to set parameters due to variance in weight and bias array sizes"
        self.weights = weights
        self.biases = biases
        self.layer_shape = (len(biases), len(weights[0]))
        return self.layer_shape