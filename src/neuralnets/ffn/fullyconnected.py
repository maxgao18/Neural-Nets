import random
import math
import functions as fn
import numpy as np

# -- Class for the neural network
class FullyConnectedNet:
    # Args:
    #   layer_sizes (list) - list specifying how many neurons each layer should have
    #   cost (cost function) optional - which cost function should be used (quad cost or cross entropy)
    #   logistic_func (squashing function) optional - which squashing function should be used for the network
    def __init__(self, layer_sizes, cost=fn.CrossEntropy, logistic_func=fn.Sigmoid):
        self.__layer_sizes = layer_sizes
        self.__n_layers = len(layer_sizes)

        # Define cost function
        self.__cost = cost

        # Define logistic function
        self.__logistic_func = logistic_func

        # Initialize biases by Gaussian/Normal distribution with mean = 0 and std_dev = 1
        # Initialize weights by Gaussian/Normal distribution with mean = 0 and std_dev = 1/sqrt(layer_size)

        # Biases is a 2D array with each layers biases. The input and output layer have no biases
        self.__biases = [np.random.randn(l) for l in layer_sizes[1:]]
        # Weights is a 3D array w[x][y][z] where x is the layer number, y is the neuron on layer x, and z is the weight
        # connecting neuron z on layer x-1 to neuron y in layer x
        self.__weights = [np.random.randn(cl, pl)/np.sqrt(cl) for pl, cl in zip(self.__layer_sizes[:self.__n_layers-1], self.__layer_sizes[1:])]

    # Returns next activation, the z value
    # Args:
    #   curr_activations (np array) - a 1D np array of the current activations
    #   curr_layer (int) - the current layer number (-1) (actual_curr_layer-1)
    def __next_activation(self, curr_activations, curr_layer):
        return np.dot(self.__weights[curr_layer], curr_activations) + self.__biases[curr_layer]

    # Returns output of neural net given an input
    # Args:
    #   input_layer (np array) - 1D np array of inputs
    def feed_forward (self, input_layer):
        for l in range(0, self.__n_layers-1):
            input_layer = self.__logistic_func.func(self.__next_activation(input_layer, l))
        return input_layer

    # Returns nabla_b and nabla_w, gradients of biases and weights by back propagation
    # Error (l) = hadamard(weights_transpose(l+1)*error(l+1), sig_prime(activations(l))
    # Args:
    #   network_input (np array) - the input for the network
    #   expected_out (np array) - the expected output for that input
    def __back_prop(self, network_input, expected_out):
        z = network_input
        z_s = [network_input]
        # 2D arrays storing activations of each neuron on each layer
        activation_vecs = [network_input]
        activation_vecs_prime = [np.zeros(self.__layer_sizes[0])]
        for l in range(0, self.__n_layers-1):
            z = self.__next_activation(z, l)
            z_s.append(z)
            activation_vecs.append(self.__logistic_func.func(z))
            activation_vecs_prime.append(self.__logistic_func.func_deriv(z))
            z = self.__logistic_func.func(z)
        # Gradient of biases are a 2D array and weights are a 3D array
        # Gradient of biases is a 2D array b[l][n] storing the bias of layer l-1 and neuron n-1
        grad_b = [np.zeros(b.shape) for b in self.__biases]
        # Gradient of weights is a 3D array w[l][n][p] storing the weight
        #   connecting neuron p-1 on layer l-1 to neuron n-1 on layer l
        grad_w = [np.zeros(w.shape) for w in self.__weights]

        # Initial error hadamard(d_cost, sig_prime)
        error = self.__cost.delta(activation_vecs[self.__n_layers-1], expected_out, z_s[self.__n_layers-1])

        grad_b[self.__n_layers-2] = error
        grad_w[self.__n_layers-2] = np.dot(np.array([error]).transpose(), np.array([activation_vecs[self.__n_layers-2]]))

        for l in reversed(range(1, self.__n_layers-1)):
            error = np.dot(self.__weights[l].transpose(), error)*activation_vecs_prime[l]
            grad_b[l-1] = error
            grad_w[l-1] = np.dot(np.array([error]).transpose(), np.array([activation_vecs[l-1]]))
        return grad_b, grad_w

    # Updates networks weights and biases based on gradients, lambda, and size of training set through regularization
    # Args:
    #   mini_batch (list) - the mini batch to be used to update the network
    #   step_size (float) - the step size, which determines how much the weights and biases should be modified
    #   lmbda (float) - a regularization variable
    #   training_set_size (int) - the total number of training inputs in the training set
    #   regularization_type (string) optional - which regularization should be used (None, "L1", "L2")
    def __update_net_weights_biases (self, mini_batch, step_size, lmbda, training_set_size, regularization_type=None):
        grad_b = [np.zeros(l) for l in self.__layer_sizes[1:]]
        grad_w = [np.zeros((cl, pl)) for pl, cl in zip(self.__layer_sizes[:self.__n_layers-1], self.__layer_sizes[1:])]

        # Calculate the gradients for the mini-batch
        for i, o in mini_batch:
            d_grad_b, d_grad_w = self.__back_prop(i, o)
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, d_grad_b)]
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, d_grad_w)]

        # Since a "mini_batch_size" number of gradients are calculated, multiply by step size and divide by the number
        # of cases (average gradient)
        avg_step = step_size/(len(mini_batch)+0.0)

        # No regularization for biases
        self.__biases = [b - avg_step*gb for b, gb in zip(self.__biases, grad_b)]

        if regularization_type == 'L1':
            #L1 Regularization
            self.__weights = [w - avg_step*gw for w, gw in zip(self.__weights, grad_w)]

            reg = lmbda/(training_set_size+0.0)
            for x in range(len(self.__weights)):
                for y in range(len(self.__weights[x])):
                    for z in range(len(self.__weights[x][y])):
                        self.__weights[x][y][z] -= fn.sign(self.__weights[x][y][z])*reg
        elif regularization_type == 'L2':
            #L2 Regularization
            reg = lmbda/(training_set_size+0.0)
            self.__weights = [(1-step_size*reg)*w-step_size*gw for w, gw in zip (self.__weights, grad_w)]
        else:
            # Unregularized
            self.__weights = [w-avg_step*gw for w, gw in zip(self.__weights, grad_w)]

    # Returns the fraction of test cases correctly guessed by the neural net for "test_data"
    # Args:
    #   test_data (list of tuples) - a list of tuples (network_input, expected_output)
    def evaluate(self, test_data):
        n_tests = len(test_data)
        tests = [0 for x in range(n_tests)]
        expected = [0 for x in range(n_tests)]

        for a in range(n_tests):
            tests[a] = test_data[a][0]
            expected[a] = test_data[a][1]
        results = []

        for t in range(n_tests):
            results.append(self.feed_forward(tests[t]))
        correct = 0.0

        for t in range(n_tests):
            max = -1
            ind = 0
            for c in range(0, len(results[t])):
                if results[t][c] > max:
                    max = results[t][c]
                    ind = c
            res = 0
            for c in range(1, len(expected[t])):
                if expected[t][c] == 1:
                    res = c
            if res == ind:
                correct += 1

        return correct / n_tests

    # Performs SGD to network
    # Args:
    #   epochs - number of times to train network with on the entire training set
    #   mini_batch_size - how large each random batch of test cases should be before performing back propagation
    #   training_inputs - a list of inputs (1D vectors) for the network
    #   expected_outputs - a list of expected outputs (1D vectors) for the network, in the order of th training inputs
    #   step_size - step size to be used while performing SGD
    def stochastic_gradient_descent(self, epochs, mini_batch_size, training_inputs, expected_outputs,
                                    step_size, lmbda=0, regularization_type=None, test_input=None, test_output=None):
        # Bind input with its expected output
        training_set_size = len(training_inputs)
        training_data = []
        for t in range(len(training_inputs)):
            training_data.append([training_inputs[t], expected_outputs[t]])
        if len(training_data) == 0:
            return

        # If validation data exists, do the same with the validation set
        test_data = []
        if test_input:
            for i, o in zip(test_input, test_output):
                test_data.append([i, o])

        # Perform SGD
        for iters in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[curr:curr+mini_batch_size] for curr in
                            range(0, len(training_data), mini_batch_size)]
            for batch in mini_batches:
                self.__update_net_weights_biases(batch, step_size, lmbda, training_set_size, regularization_type)

            if test_input:
                print("Epoch:", iters+1, ", Percent correct:", self.evaluate(test_data))
            else:
                print("Epoch:", iters+1, ", Percent correct:", self.evaluate(mini_batches[0]))

    # Returns all weights in the neural network (3D Array)
    def get_weights(self):
        return self.__weights

    # Returns all biases in the neural network (2D Array)
    def get_biases(self):
        return self.__biases

    # Returns the layer sizes for the network
    def get_layer_sizes(self):
        return self.__layer_sizes

    # Sets the network objects weights and biases as long as they are formatted correctly
    # Returns failure message if improper array sizes or returns 1D array of layer sizes if
    # the loading of the network is successful
    def set_weights_biases (self, weights, biases):
        if not len(weights) == len(biases):
            return "Failed to set network due to improper weight and bias array sizes"
        n_layers = len(weights) + 1
        layer_sizes = []
        layer_sizes.append(len(weights[0][0]))
        for w in weights[0]:
            if not len(w) == layer_sizes[0]:
                return "Failed to set network due to improper weight and bias array sizes"
        for l, b, n in zip(weights, biases, range(0, n_layers-1)):
            if len(l) == len(b):
                layer_sizes.append(len(b))
                for w in l:
                    if not len(w) == layer_sizes[n]:
                        return "Failed to set network due to improper weight and bias array sizes"
            else:
                return "Failed to set network due to improper weight and bias array sizes"
        self.__n_layers = n_layers
        self.__weights = weights
        self.__biases = biases
        self.__layer_sizes = layer_sizes
        return layer_sizes