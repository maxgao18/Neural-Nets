import numpy as np
from kernel import Kernel
from layer import Layer

from functions import LeakyRELU
from functions import RELU

class ConvLayer(Layer):
    # Args:
    #   input_shape (3 tuple (ints)) - (input depth, input height, input length)
    #   kernel_shape (4 tuple (ints)) - (num kernels, kernel depth, kernel height, kernel length)
    def __init__(self, input_shape, kernel_shape, kernels=None, activation_function=RELU):
        super(ConvLayer, self).__init__(input_shape=input_shape,
                                        output_shape=(kernel_shape[0],
                                                      input_shape[1]-kernel_shape[2]+1,
                                                      input_shape[2]-kernel_shape[3]+1),
                                        activation_function=activation_function)
        self.kernel_shape = kernel_shape

        if kernels is not None:
            self.kernels = kernels
        else:
            self.kernels = []
            for x in range(kernel_shape[0]):
                self.kernels.append(Kernel(kernel_shape[1:]))


    # Similar to feedforward, but without squashing
    # Args: image (3D np arr) - the image
    def getactivations(self, inputs):
        new_img = []
        for k in self.kernels:
            new_img.append(k.use_kernel(inputs))
        return np.array(new_img)

    # Returns the new image created using the current layers kernels squashed by an activation function
    # Args: image (3D np arr) - the image
    def feedforward(self, inputs):
        new_img = self.getactivations(inputs)
        return self.activation_function.func(new_img)

    # Returns the kernel errors (weights and biases) and the previous image error
    # Args:
    #   z-activations (3D np arr) - activations for the previous layer
    #   deltas (3D np arr) - the errors in the forward layer
    def backprop (self, prev_fz_activations, d_prev_z_activations, curr_deltas):
        prevDeltas = np.zeros(self.input_shape)
        kernelWeightDeltas = []
        kernelBiasDeltas = []

        for k, d in zip(self.kernels, curr_deltas):
            wd, bd, pd = k.backprop(self.input_shape, self.output_shape, prev_fz_activations, d_prev_z_activations, d)
            prevDeltas+=pd
            kernelWeightDeltas.append(wd)
            kernelBiasDeltas.append(bd)

        return np.array(kernelWeightDeltas), np.array(kernelBiasDeltas), prevDeltas

    def getdeltas(self, d_prev_z_activations, curr_deltas):
        prevDeltas = np.zeros(self.input_shape)
        for k, d in zip(self.kernels, curr_deltas):
            prevDeltas += k.getdeltas(self.input_shape, self.output_shape, d_prev_z_activations, d)

        return prevDeltas

    # Update the kernels
    # Args:
    #   d_weights (4D np arr) - amount to change kernel weights
    #   d_bias (1D np arr) - amount to change kernel biases
    def update (self, d_weights, d_bias):
        for i, k in enumerate(self.kernels):
            k.update(d_weights[i], d_bias[i])

    def get_kernels(self, index=-1):
        if index == -1:
            return self.kernels
        return self.kernels[index]
