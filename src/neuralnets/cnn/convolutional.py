import numpy as np

from layers import Kernel

from layers import ConvLayer
from layers import DeconvLayer
from layers import DenseLayer
from layers import SoftmaxLayer

from functions import QuadraticCost
from functions import NegativeLogLikelihood

from functions import LeakyRELU
from functions import Softmax

from random import shuffle
from copy import deepcopy

# Makes a 3D np array into a 1D np array
def flatten_image(image):
    l = np.array([])
    for x in image:
        l = np.concatenate((l, x.ravel()))

    image = l.ravel()
    return image

# Makes 1D np array into 3D np array
def convert_to_image(arr, image_shape):
    image = np.zeros(image_shape)
    counter = 0

    for z in range(image_shape[0]):
        for y in range(image_shape[1]):
            for x in range(image_shape[2]):
                image[z][y][x] = arr[counter]
                counter+=1

    return image

class ConvolutionalNet:
    # Args:
    #   input_shape (tuple) - the shape of the input (for images: (image depth, image height, image length))
    def __init__(self, input_shape, layers=None, cost_func=NegativeLogLikelihood):
        self.input_shape = input_shape
        self.layer_types = []
        self.num_layers = 0
        self.cost_func = cost_func
        self.layers = []
        if layers is not None:
            self.layers = layers

    # Adds a new layer to the network
    # Args:
    #   layer_type (string) - the type of layer to be added (conv, deconv, dense, soft)
    #   output_size (tuple/int) optional - the shape of the output for that layer
    #                   conv (None)
    #                   deconv (2 tuple): (output height, output length)
    #                   dense and softmax (int): num of neurons on the layer
    #   kernel_size (2-tuple) optional - for conv and deconv layers, (num kernels, kernel height, kernel length)
    def add(self, layer_type, output_size=None, kernel_size=None):
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
                    num_prev_neurons = input_shape[0]*input_shape[1]*input_shape[2]

            # Shape for layers are (num neurons, num prev neurons)
            layer_shape = (output_size, num_prev_neurons)
            if layer_type is "dense":
                self.layers.append(DenseLayer(layer_shape=layer_shape))
            elif layer_type is "soft":
                self.layers.append(SoftmaxLayer(layer_shape=layer_shape))

        self.num_layers += 1
        self.layer_types.append(layer_type)

    # Returns the next activation without squashing it
    # Args:
    #   z_activations - (np arr) the current activations
    #   layer - the next layer to be used
    def next_activation(self, z_activations, layer):
        return layer.get_activations(z_activations)

    # Feeds an input through the network, returning the output
    # Args: network_input - (np arr) the input
    def feed_forward(self, network_input):
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

            network_input = lyr.feed_forward(network_input)

        return network_input

    # This function calculates the gradients for one training example
    # Args:
    #   network_input - (np arr) the input being used
    #   expected_output - (np arr) the expected output
    def backprop(self, network_input, expected_output):
        curr_z = network_input
        z_activations = [network_input]

        is_conv = False
        if self.layer_types[0] is "conv" or self.layer_types[0] is "deconv":
            is_conv = True

        for i, lt, lyr in zip(range(1, self.num_layers+1), self.layer_types, self.layers):
            # Squash to 1D np array
            if lt is not "conv" and lt is not "deconv" and is_conv:
                is_conv = False
                curr_z = flatten_image(curr_z)

            curr_z = lyr.get_activations(curr_z)
            z_activations.append(deepcopy(curr_z))

            if not i == self.num_layers:
                # Use softmax for SM layers, otherwise leaky relu
                if lt is "soft":
                    curr_z = Softmax.func(curr_z)
                else:
                    curr_z = LeakyRELU.func(curr_z)

        # Store derivatives and activation for output layer
        if self.layer_types[-1] is "soft":
            squashed_activations = Softmax.func(deepcopy(curr_z))
            squashed_activations_deriv = Softmax.func_deriv(deepcopy(curr_z))
        else:
            squashed_activations = LeakyRELU.func_deriv(deepcopy(curr_z))
            squashed_activations_deriv = LeakyRELU.func_deriv(deepcopy(curr_z))

        # Errors for the last layer
        delta = self.cost_func.delta(squashed_activations,
                                     squashed_activations_deriv,
                                     expected_output)

        is_conv = True
        if self.layer_types[self.num_layers-1] is not "conv" \
                and self.layer_types[self.num_layers-1] is not "deconv":
            is_conv = False

        delta_w = []
        delta_b = []

        # Append all the errors for each layer
        for lt, lyr, zprev in reversed(zip(self.layer_types, self.layers, z_activations[:-1])):
            if lt is "conv" or lt is "deconv":
                if not is_conv:
                    delta = convert_to_image(delta, lyr.get_output_shape())
                    is_conv = True

            dw, db, dlt = lyr.backprop(zprev, delta)
            delta_w.insert(0, dw)
            delta_b.insert(0, db)

            delta = dlt

        return np.array(delta_w), np.array(delta_b)

    # Updates the network given a specific minibatch (done by averaging gradients over the minibatch)
    # Args:
    #   mini_batch - a list of tuples, (input, expected output)
    #   step_size - the amount the network should change its parameters by relative to the gradients
    def update_network(self, mini_batch, step_size):
        gradient_w, gradient_b = self.backprop(mini_batch[0][0], mini_batch[0][1])

        for inp, outp in mini_batch[1:]:
            dgw, dgb = self.backprop(inp, outp)
            gradient_w += dgw
            gradient_b += dgb

        # Average the gradients
        gradient_w *= step_size/(len(mini_batch)+0.00)
        gradient_b *= step_size/(len(mini_batch)+0.00)

        # Update weights and biases in opposite direction of gradients
        for gw, gb, lyr in zip(gradient_w, gradient_b, self.layers):
            lyr.update(-gw, -gb)

    # Evaluates the average cost across the training set
    def evaluate_cost(self, training_set):
        total = 0.0
        for inp, outp in training_set:
            net_outp = self.feed_forward(inp)
            total += self.cost_func.cost(net_outp, outp)
        return total/len(training_set)

    # Performs SGD on the network
    # Args:
    #   epochs - (int), number of times to loop over the entire batch
    #   step_size - (float), amount network should change its parameters per update
    #   mini_batch_size - (int), number of training examples per mini batch
    #   training_inputs - (list), the list of training inputs
    #   expected_outputs - (list), the list of expected outputs for each input
    def stochastic_gradient_descent(self, epochs, step_size, mini_batch_size, training_inputs, expected_outputs):
        training_set = []
        for inp, outp in zip(training_inputs, expected_outputs):
            training_set.append((inp, outp))

        # Train
        for ep in range(epochs):
            shuffle(training_set)
            for x in range(0, len(training_set), mini_batch_size):
                self.update_network(training_set[x:x+mini_batch_size], step_size)
            # Update with progress
            print("Epoch: %d   Average cost: %f" % (ep+1, self.evaluate_cost(training_set)))
