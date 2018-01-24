from layers import SoftmaxLayer
from layers import RecurrentLayer

from functions import NegativeLogLikelihood
from functions import Softmax
from functions import LeakyRELU

from copy import deepcopy
import numpy as np
from random import shuffle

class RecurrentNet:
    def __init__(self, num_inputs, layers=None, cost_func=NegativeLogLikelihood):
        self.num_inputs = num_inputs
        self.num_layers = 0
        self.layer_types = []

        if layers is not None:
            self.layers = layers
            self.num_layers = len(self.layers)

            for l in layers:
                if isinstance(l, SoftmaxLayer):
                    self.layer_types.append("soft")
                elif isinstance(l, RecurrentLayer):
                    self.layer_types.append("recurr")
        else:
            self.layers = []

        self.cost_func = cost_func

    def add(self, layer_type, output_size):
        op = self.num_inputs
        if len(self.layers) > 0:
            op = self.layers[-1].get_output_shape()

        layer_shape = (output_size, op)
        if layer_type is "soft":
            self.layers.append(SoftmaxLayer(layer_shape))
        elif layer_type is "recurr":
            self.layers.append(RecurrentLayer(layer_shape))

        self.layer_types.append(layer_type)

        self.num_layers+=1

    def forget_past(self):
        for l in self.layers:
            if isinstance(l, RecurrentLayer):
                l.forget_past()

    def feed_forward(self, network_input):
        for l in self.layers:
            network_input = l.feed_forward(network_input)
        return network_input

    def backprop(self, network_input, expected_output):
        curr_z = network_input
        z_activations = [network_input]
        p_z_activations = []

        for i, lt, lyr in zip(range(1, self.num_layers + 1), self.layer_types, self.layers):
            if lt == "recurr":
                prev_z, curr_z = lyr.get_activations(curr_z)
                z_activations.append(deepcopy(curr_z))
                p_z_activations.append(prev_z)
            elif lt == "soft":
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
        if self.layer_types[self.num_layers - 1] is not "conv" \
                and self.layer_types[self.num_layers - 1] is not "deconv":
            is_conv = False

        delta_w = []
        delta_pw = []
        delta_b = []

        cnt = -1
        # Append all the errors for each layer
        for i, lt, lyr, zprev in reversed(zip(range(self.num_layers), self.layer_types, self.layers, z_activations[:-1])):
            if lt is "soft":
                dw, db, dlt = lyr.backprop(zprev, delta)
                delta_w.insert(0, dw)
                delta_b.insert(0, db)

                delta = dlt
            elif lt is "recurr":
                dw, dpw, db, dlt = lyr.backprop(p_z_activations[cnt], zprev, delta)
                delta_w.insert(0, dw)
                delta_pw.insert(0, dpw)
                delta_b.insert(0, db)

                delta = dlt

                cnt-=1

        return np.array(delta_w), np.array(delta_pw), np.array(delta_b)

    # Updates the network given a specific minibatch (done by averaging gradients over the minibatch)
    # Args:
    #   mini_batch - a list of tuples, (input, expected output)
    #   step_size - the amount the network should change its parameters by relative to the gradients
    def update_network(self, mini_batch, step_size):
        recurrent_indicies = [False for i in range(self.num_layers)]
        for i, l in enumerate(self.layers):
            if isinstance(l, RecurrentLayer):
                recurrent_indicies[i] = True

        gradient_w, gradient_pw, gradient_b = self.backprop(mini_batch[0][0], mini_batch[0][1])

        for inp, outp in mini_batch[1:]:
            dgw, dgpw, dgb = self.backprop(inp, outp)
            gradient_w += dgw
            gradient_pw += dgpw
            gradient_b += dgb

        # Average the gradients
        gradient_w *= step_size / (len(mini_batch) + 0.00)
        gradient_pw *= step_size / (len(mini_batch) + 0.00)
        gradient_b *= step_size / (len(mini_batch) + 0.00)

        cnt = 0
        # Update weights and biases in opposite direction of gradients
        for i, gw, gb, lyr in zip(range(self.num_layers), gradient_w, gradient_b, self.layers):
            if recurrent_indicies[i]:
                lyr.update(-gw, -gradient_pw[cnt], -gb)
                cnt+=1
            else:
                lyr.update(-gw, -gb)

    # Evaluates the average cost across the training set
    def evaluate_cost(self, training_set):
        total = 0.0
        for inp, outp in training_set:
            net_outp = self.feed_forward(inp)
            total += self.cost_func.cost(net_outp, outp)
        return total / len(training_set)

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
            for x in range(0, len(training_set), mini_batch_size):
                self.update_network(training_set[x:x + mini_batch_size], step_size)
            # Update with progress
            print("Epoch: %d   Average cost: %f" % (ep + 1, self.evaluate_cost(training_set)))
            self.forget_past()
