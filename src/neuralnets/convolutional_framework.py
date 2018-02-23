from layers import Kernel

from layers import ConvLayer
from layers import DeconvLayer
from layers import DenseLayer
from layers import SoftmaxLayer

from functions import QuadraticCost
from functions import NegativeLogLikelihood

from neural_network import NeuralNetwork

from abc import abstractmethod

import numpy as np

# Makes a 3D np array into a 1D np array
def flatten_image(image):
    if len(image.shape) > 1:
        l = np.array([])
        for x in image:
            l = np.concatenate((l, x.ravel()))

        image = l.ravel()
    return image

# Makes 1D np array into 3D np array
def convert_to_image(arr, image_shape):
    if len(arr.shape) == 2:
        return np.array([arr])
    elif len(arr.shape) < 2:
        image = np.zeros(image_shape)
        counter = 0

        for z in range(image_shape[0]):
            for y in range(image_shape[1]):
                for x in range(image_shape[2]):
                    image[z][y][x] = arr[counter]
                    counter+=1
        return image
    return arr


class ConvolutionalFramework(NeuralNetwork):
    def __init__(self, network_type, cost_function, layers=None):
       super(ConvolutionalFramework, self).__init__(network_type, cost_function, layers)

    # Adds a new layer to the network
    # Args:
    #   layer_type (string) - the type of layer to be added (conv, deconv, dense, soft)
    #   output_size (tuple/int) optional - the shape of the output for that layer
    #                   conv (None)
    #                   deconv (2 tuple): (output height, output length)
    #                   dense and softmax (int): num of neurons on the layer
    #   kernel_size (2-tuple) optional - for conv and deconv layers, (num kernels, kernel height, kernel length)
    def addlayer(self, layer_type, output_size=None, kernel_size=None):
       # If there are no layers, make the first one
       input_shape = self.input_shape
       is_first_layer = True

       # Use last layer as input shape for new layer if there exists a previous layer
       if not len(self.layers) == 0:
           input_shape = self.layers[-1].get_output_shape()
           is_first_layer = False

       if layer_type is "conv" or layer_type is "deconv":
           # Order kernel shape (num kernels, kernel depth, kernel height, kernel length)
           kernel_shape = (kernel_size[0], input_shape[0], kernel_size[1], kernel_size[2])

           if layer_type is "conv":
               self.layers.append(ConvLayer(input_shape=input_shape,
                                            kernel_shape=kernel_shape))
           elif layer_type is "deconv":
               # Order output shape (image depth, image height, image length)
               output_shape = (kernel_size[0], output_size[0], output_size[1])
               self.layers.append(DeconvLayer(input_shape=input_shape,
                                              output_shape=output_shape,
                                              kernel_shape=kernel_shape))
       elif layer_type is "dense" or layer_type is "soft":
           # Assume last layer was softmax or dense
           num_prev_neurons = input_shape
           # If it is a deconv or conv, calculate number of previous neurons
           if not is_first_layer:
               if self.layer_types[-1] is "conv" or self.layer_types[-1] is "deconv":
                   num_prev_neurons = input_shape[0] * input_shape[1] * input_shape[2]

           if layer_type is "dense":
               self.layers.append(DenseLayer(input_shape=num_prev_neurons,
                                             output_shape=output_size))
           elif layer_type is "soft":
               self.layers.append(SoftmaxLayer(input_shape=num_prev_neurons,
                                               output_shape=output_size))

       self.num_layers += 1
       self.layer_types.append(layer_type)

       # Change cost function based on last layer to optimize training
       if isinstance(self.layers[-1], SoftmaxLayer):
           self.cost_function = NegativeLogLikelihood
       else:
            self.cost_function = QuadraticCost

    # Feeds an input through the network, returning the output
    # Args: network_input - (np arr) the input
    def feedforward(self, network_input):
        is_conv = False
        if self.layer_types[0] == "conv" or self.layer_types[0] == "deconv":
            is_conv = True
            if len(network_input.shape) == 2:
                network_input = np.array([network_input])


        for lt, lyr in zip(self.layer_types, self.layers):
            # Squash to 1D np array
            if lt is not "conv" and lt is not "deconv" and is_conv:
                is_conv = False
                network_input = flatten_image(network_input)

            network_input = lyr.feedforward(network_input)

        return network_input
