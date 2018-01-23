import numpy as np
from kernel import Kernel

from functions import LeakyRELU

# leaky relu function
def func (z):
    return LeakyRELU.func(z)

def func_deriv(z):
    return LeakyRELU.func_deriv(z)

class ConvLayer:
    # Args:
    #   input_shape (3 tuple (ints)) - (input depth, input height, input length)
    #   kernel_shape (4 tuple (ints)) - (num kernels, kernel depth, kernel height, kernel length)
    def __init__(self, input_shape, kernel_shape, kernels=None):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.output_shape = (kernel_shape[0], input_shape[1]-kernel_shape[2]+1, input_shape[2]-kernel_shape[3]+1)

        if kernels is not None:
            self.kernels = kernels
        else:
            self.kernels = []
            for x in range(kernel_shape[0]):
                self.kernels.append(Kernel(kernel_shape[1:]))

    def get_kernels(self, index=-1):
        if index == -1:
            return self.kernels
        return self.kernels[index]


    # Similar to feedforward, but without squashing
    # Args: image (3D np arr) - the image
    def get_activations(self, image):
        new_img = []
        for k in self.kernels:
            new_img.append(k.use_kernel(image))
        return np.array(new_img)

    # Returns the new image created using the current layers kernels squashed by an activation function
    # Args: image (3D np arr) - the image
    def feed_forward(self, image):
        new_img = self.get_activations(image)
        return func(new_img)

    # Returns the kernel errors (weights and biases) and the previous image error
    # Args:
    #   z-activations (3D np arr) - activations for the previous layer
    #   deltas (3D np arr) - the errors in the forward layer
    def backprop (self, z_activations, deltas):
        prevDeltas = np.zeros(self.input_shape)
        kernelWeightDeltas = []
        kernelBiasDeltas = []

        for k, d in zip(self.kernels, deltas):
            wd, bd, pd = k.get_errors(self.input_shape, self.output_shape, z_activations, d)
            prevDeltas+=pd
            kernelWeightDeltas.append(wd)
            kernelBiasDeltas.append(bd)

        return np.array(kernelWeightDeltas), np.array(kernelBiasDeltas), prevDeltas

    # Update the kernels
    # Args:
    #   d_weights (4D np arr) - amount to change kernel weights
    #   d_bias (1D np arr) - amount to change kernel biases
    def update (self, d_weights, d_bias):
        for i, k in enumerate(self.kernels):
            k.update(d_weights[i], d_bias[i])

    def get_output_shape(self):
        return self.output_shape
