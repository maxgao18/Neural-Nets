import numpy as np
from functions import Softmax
from dense_layer import DenseLayer

class SoftmaxLayer(DenseLayer):
    # Args:
    #   layer_shape - a 2-tuple of ints (number of neurons on current layer, number of neurons on previous layer)
    #   weights (optional) - a 2D np array of the weights
    #   biases (optional) a 1D np array of the biases
    def __init__(self, input_shape, output_shape, weights=None, biases=None):
        super(SoftmaxLayer,self).__init__(input_shape=input_shape,
                                          output_shape=output_shape,
                                          weights=weights,
                                          biases=biases,
                                          activation_function=Softmax)