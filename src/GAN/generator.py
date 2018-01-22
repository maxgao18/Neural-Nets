import numpy as np

from kernel import Kernel

from conv_layer import ConvLayer
from deconv_layer import DeconvLayer
from dense_layer import DenseLayer
from softmax_layer import SoftmaxLayer

from cost_functions import QuadCost
from cost_functions import NegLogLikehood

from activation_functions import LeakyRELU
from activation_functions import Softmax

from random import shuffle
from copy import deepcopy

from discriminator import Discriminator

class Generator:
    # Args:
    #   input_shape (tuple) - the shape of the input (for images: (image depth, image height, image length))
    def __init__(self, input_shape, layers=None):
        self.input_shape = input_shape
        self.layer_types = []
        self.num_layers = 0
        self.layers = []
        if layers is not None:
            self.layers = layers

    # Adds a new layer to the network
    # Args:
    #   layer_type (string) - the type of layer to be added (conv, deconv, dense, soft)
    #   output_size (tuple/int) optional - the shape of the output for that layer
    #                   conv (None)
    #                   deconv (2 tuple): (output height, output length)
    #   kernel_size (2-tuple) - for conv and deconv layers, (num kernels, kernel height, kernel length)
    def add(self, layer_type, output_size, kernel_size):
        # If there are no layers, make the first one
        input_shape = self.input_shape

        # Use last layer as input shape for new layer if there exists a previous layer
        if not len(self.layers) == 0:
            input_shape = self.layers[-1].get_output_shape()

        # Order kernel shape (num kernels, kernel depth, kernel height, kernel length)
        kernel_shape = (kernel_size[0], input_shape[0], kernel_size[1], kernel_size[2])

        if layer_type is "conv":
            self.layers.append(ConvLayer(image_shape=input_shape,
                                         kernel_shape=kernel_shape))
        elif layer_type is "deconv":
            # Order output shape (image depth, image height, image length)
            output_shape = (kernel_size[0], output_size[0], output_size[1])
            self.layers.append(DeconvLayer(input_shape=input_shape,
                                           output_shape=output_shape,
                                           kernel_shape=kernel_shape))
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
        if len(network_input.shape) == 2:
            network_input = np.array([network_input])

        for lyr in self.layers:
            network_input = lyr.feed_forward(network_input)

        return network_input

    # This function calculates the gradients for one training example
    # Args:
    #   network_input - (np arr) the input being used
    #   discriminator_network (object)
    def backprop(self, network_input, expected_output, discriminator_network):
        curr_z = network_input
        z_activations = [network_input]

        for i, lyr in zip(range(1, self.num_layers+1), self.layers):
            curr_z = lyr.get_activations(curr_z)
            z_activations.append(deepcopy(curr_z))
            curr_z = LeakyRELU.func(curr_z)


        delta = discriminator_network.get_delta(deepcopy(curr_z), expected_output)
        delta_w = []
        delta_b = []

        # Append all the errors for each layer
        for lyr, zprev in reversed(zip(self.layers, z_activations[:-1])):
            dw, db, dlt = lyr.backprop(zprev, delta)
            delta_w.insert(0, dw)
            delta_b.insert(0, db)

            delta = dlt

        return np.array(delta_w), np.array(delta_b)

    # Updates the network given a specific minibatch (done by averaging gradients over the minibatch)
    # Args:
    #   mini_batch - a list of np arrays (inputs)
    #   step_size - the amount the network should change its parameters by relative to the gradients
    def update_network(self, mini_batch, step_size, discriminator_network):
        gradient_w, gradient_b = self.backprop(mini_batch[0][0], mini_batch[0][1], discriminator_network)

        for inp, outp in mini_batch[1:]:
            dgw, dgb = self.backprop(inp, outp, discriminator_network)
            gradient_w += dgw
            gradient_b += dgb

        # Average the gradients
        gradient_w *= step_size/(len(mini_batch)+0.00)
        gradient_b *= step_size/(len(mini_batch)+0.00)

        # Update weights and biases in opposite direction of gradients
        for gw, gb, lyr in zip(gradient_w, gradient_b, self.layers):
            lyr.update(-gw, -gb)

    # Evaluates the average cost across the training set
    def evaluate_cost(self, training_set, discriminator_network):
        total = 0.0
        for inp, outp in training_set:
            net_outp = self.feed_forward(inp)
            total += discriminator_network.cost_func.cost(discriminator_network.feed_forward(net_outp), outp)

        return total/len(training_set)

    # Performs SGD on the network
    # Args:
    #   epochs - (int), number of times to loop over the entire batch
    #   step_size - (float), amount network should change its parameters per update
    #   mini_batch_size - (int), number of training examples per mini batch
    #   training_inputs - (list), the list of training inputs
    #   expected_outputs - (list), the list of expected outputs for each input
    def stochastic_gradient_descent(self, epochs, step_size, mini_batch_size, training_set, discriminator_network):
        # Train
        for ep in range(epochs):
            shuffle(training_set)
            for x in range(0, len(training_set), mini_batch_size):
                self.update_network(training_set[x:x+mini_batch_size], step_size, discriminator_network)
            # Update with progress
            print("Generator Epoch: %d   Average cost: %f" % (ep+1, self.evaluate_cost(training_set, discriminator_network)))