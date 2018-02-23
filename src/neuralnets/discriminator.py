import numpy as np

from layers import Kernel

from layers import ConvLayer
from layers import DeconvLayer
from layers import DenseLayer
from layers import SoftmaxLayer

from functions import QuadraticCost
from functions import NegativeLogLikelihood

from convolutional import ConvolutionalNet

from random import shuffle
from copy import deepcopy

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

class Discriminator (ConvolutionalNet):
    # Args:
    #   input_shape (tuple) - the shape of the input (for images: (image depth, image height, image length))
    def __init__(self, input_shape, layers=None, cost_func=QuadraticCost):
        super(Discriminator, self).__init__(input_shape, layers, cost_func)

    def getdeltas(self, network_input, expected_output):
        curr_z = network_input
        dzs_list = [network_input]

        is_conv = False
        if self.layer_types[0] is "conv" or self.layer_types[0] is "deconv":
            is_conv = True

        for i, lt, lyr in zip(range(1, self.num_layers + 1), self.layer_types, self.layers):
            # Squash to 1D np array
            if lt is not "conv" and lt is not "deconv" and is_conv:
                is_conv = False
                curr_z = flatten_image(curr_z)

            curr_z = lyr.getactivations(curr_z)
            dzs_list.append(lyr.activation_function.func_deriv(deepcopy(curr_z)))

            curr_z = lyr.activation_function.func(curr_z)

        # Errors for the last layer
        delta = self.cost_function.delta(curr_z,
                                         dzs_list[-1],
                                         expected_output)

        is_conv = True
        if self.layer_types[-1] is not "conv" \
                and self.layer_types[-1] is not "deconv":
            is_conv = False

        # Append all the errors for each layer
        for lt, lyr, dzs in reversed(zip(self.layer_types, self.layers, dzs_list[:-1])):
            if lt is "conv" or lt is "deconv":
                if not is_conv:
                    delta = convert_to_image(delta, lyr.get_output_shape())
                    is_conv = True
            elif lt is "dense" or lt is "soft":
                dzs = flatten_image(dzs)
            delta = lyr.getdeltas(dzs, delta)

        return delta

    # Performs SGD on the network
    # Args:
    #   epochs - (int), number of times to loop over the entire batch
    #   step_size - (float), amount network should change its parameters per update
    #   mini_batch_size - (int), number of training examples per mini batch
    #   training_inputs - (list), the list of training inputs
    #   expected_outputs - (list), the list of expected outputs for each input
    def stochastic_gradient_descent(self, epochs, step_size, mini_batch_size, training_set):
        # Train
        for ep in range(epochs):
            shuffle(training_set)
            for x in range(0, len(training_set), mini_batch_size):
                self.update_network(training_set[x:x+mini_batch_size], step_size)
            # Update with progress
            print("Discriminator Epoch: %d   Average cost: %f" % (ep+1, self.evaluate_cost(training_set)))
