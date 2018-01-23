from kernel import Kernel
import numpy as np

from functions import LeakyRELU

# leaky relu function
def func (z):
    return LeakyRELU.func(z)

def func_deriv(z):
    return LeakyRELU.func_deriv(z)

# Pads an image with zeros given a mapping
# Args:
#   image (3D np array) - a list of images
#   padded_image_shape (tuple) - the desired padded image shape (depth, height, length)
#   input_to_padded (dictionary) - a mapping from a 2D coordinate (input) to a 2D coordinate on the padded image
def pad(image, padded_image_shape, input_to_padded):
    padded_image = np.zeros(padded_image_shape)
    for i, img in enumerate(image):
        for incoord, outcoord in input_to_padded.items():
            padded_image[i][outcoord[0]][outcoord[1]] = img[incoord[0]][incoord[1]]
    return padded_image

# Unpads an image using the same operation as the padding method, but with opposite parameters
# Args:
#   padded_image (3D np array) - the image to be unpadded
#   input_image_shape (tuple) - the desired new image shape
#   padded_to_input (dictionary) - a mapping from a 2D coordinate on the padded image onto a
#                                    2D coordinate on the input image
def unpad(padded_image, input_image_shape, padded_to_input):
    return pad(padded_image, input_image_shape, padded_to_input)


class DeconvLayer:
    # Args:
    #   input_shape (3 tuple (ints)) - (image depth, image height, image length)
    #   output_shape (3 tuple (ints)) - the expected output image shape (same format as image_shape)
    #   kernel_shape (4 tuple (ints)) - (num kernels, kernel depth, kernel height, kernel length)
    def __init__(self, input_shape, output_shape, kernel_shape, kernels=None):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.output_shape = output_shape

        # Size of the zero padded image
        self.padded_image_shape = (input_shape[0],
                                   output_shape[1]+kernel_shape[2]-1,
                                   output_shape[2]+kernel_shape[3]-1)

        # Maps a 2D coordinate from the input to a 2D coordinate on the zero padded input
        self.input_to_padded = {}
        # Maps a 2D coordinate from the zero padded input onto a 2D coordinate onto the input image
        self.padded_to_input = {}

        # Calculate how much blank space should be left between horizontal and vertical
        # adjacent pixels for evenly spaced padding
        #
        # Math:
        #   number of white spaces total = (len/height of padded image) - (len/height of input image)
        #   number of seperations = (len/height of input image) + 1
        #   length of evenly distributed blank spaces (approx) = (total white spaces) / (number of seperations)
        space_y = (self.padded_image_shape[1]-self.input_shape[1]+0.0)/(self.input_shape[1]+1.0)
        space_x = (self.padded_image_shape[2]-self.input_shape[2]+0.0)/(self.input_shape[2]+1.0)
        for y in range(input_shape[1]):
            for x in range(input_shape[2]):
                new_x = int(np.floor(space_x*(x+1))) + x
                new_y = int(np.floor(space_y*(y+1))) + y
                new_coord = (new_y, new_x)
                old_coord = (y, x)
                # Append to dictionaries
                self.input_to_padded[old_coord] = new_coord
                self.padded_to_input[new_coord] = old_coord

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
    # Args: image - 3D np array of the image
    def get_activations(self, image):
        image = pad(image, self.padded_image_shape, self.input_to_padded)
        new_img = []
        for k in self.kernels:
            new_img.append(k.use_kernel(image))
        return np.array(new_img)

    # Returns the new image created using padding and the current layers kernels squashed by an activation function
    # Args: image - 3D np array of the image
    def feed_forward(self, image):
        new_img = self.get_activations(image)
        return func(new_img)

    # Returns the kernel errors (weights and biases) and the previous image error (3D np arr)
    # Args:
    #   z-activations (3D np arr) - activations for the previous layer
    #   deltas (3D np arr) - errors in the forward layer
    def backprop(self, z_activations, deltas):
        prevDeltas = np.zeros(self.padded_image_shape)
        kernelWeightDeltas = []
        kernelBiasDeltas = []

        for k, d in zip(self.kernels, deltas):
            z_activations = pad(z_activations, self.padded_image_shape, self.input_to_padded)
            wd, bd, pd = k.get_errors(input_image_shape=self.padded_image_shape,
                                      output_image_shape=self.output_shape,
                                      z_activations=z_activations,
                                      deltas=d)
            kernelWeightDeltas.append(wd)
            kernelBiasDeltas.append(bd)
            prevDeltas += pd

        prevDeltas = unpad(prevDeltas, self.input_shape, self.padded_to_input)

        return np.array(kernelWeightDeltas), np.array(kernelBiasDeltas), prevDeltas

    # Update the kernels
    # Args:
    #   d_weights - 4d np array to change kernel weights
    #   d_bias - 1d np array to change kernel bias
    def update(self, d_weights, d_bias):
        for i, k in enumerate(self.kernels):
            k.update(d_weights[i], d_bias[i])

    def get_output_shape(self):
        return self.output_shape