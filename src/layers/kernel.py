import numpy as np

from functions import LeakyRELU

# leaky relu function
def func (z):
    return LeakyRELU.func(z)

# leaky relu derivative
def func_deriv(z):
    return LeakyRELU.func_deriv(z)

# Returns the weight errors, bias error, and previous layer errors given a 2D image
# Args:
#   in_shape (tuple) - (original image height, original image length)
#   out_shape (tuple) - (filtered image height, filtered image length)
#   zs (2D np arr) - unsquashed activations of original image
#   weights (2D np arr) - weights of a kernel
#   bias (float) - kernel bias
#   deltas (2D np arr) - forward errors (with out_shape shape)
def prev_errors (in_shape, out_shape, zs, weights, bias, deltas):
    deltasPrev = np.zeros(in_shape)
    kernelHeight = len(weights)
    kernelLength = len(weights[0])

    weightDeltas = np.zeros((kernelLength,kernelHeight))

    # Loop for each step
    for y in range(out_shape[0]):
        for x in range(out_shape[1]):
            d = deltas[y][x]

            deltasPrev[y:y+kernelHeight,x:x+kernelLength] += d*np.multiply(weights,func_deriv(zs[y:y+kernelHeight,x:x+kernelLength]))

            # --- Old method
            # # Loop for each part of the kernel
            # for wy, dpy in enumerate(range(y, y+kernelHeight)):
            #     for wx, dpx in enumerate(range(x, x+kernelLength)):
            #         deltasPrev[dpy][dpx] += d*weights[wy][wx]*func_deriv(zs[dpy][dpx])

    # Loop kernel across image to calculate grad_w
    for y in range(out_shape[0]-kernelHeight+1):
        for x in range(out_shape[1]-kernelLength+1):
            weightDeltas += np.multiply(zs[y:y+kernelHeight, x:x+kernelLength], deltas[y:y+kernelHeight, x:x+kernelLength])

            #--- Old method
            # # Calculate for one convolution
            # for wy in range(kernelHeight):
            #     for wx in range(kernelLength):
            #         weightDeltas[wy][wx] += func(zs[y+wy][x+wx])*deltas[y+wy][x+wx]

    biasDelta = np.sum(deltas)

    return weightDeltas, biasDelta, deltasPrev

# Individual kernel objects
class Kernel:
    # Args:
    #   kernel_size: a 3-tuple (kernel depth, kernel height, kernel length)
    #   weights (optional): a 3D np array of the kernels weights
    #   bias (optional): the kernels bias
    def __init__(self, kernel_size, weights=None, bias=None):
        self.kernel_size = kernel_size
        self.feature_map_length = kernel_size[2]
        self.feature_map_height = kernel_size[1]
        self.num_feature_maps = kernel_size[0]
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [np.random.randn(self.feature_map_height, self.feature_map_length) for f in range(self.num_feature_maps)]
            self.weights /= np.sqrt(self.feature_map_length*self.feature_map_height)

        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.random()

    # Takes in a list of images and applies the kernel specific to the object to the kernel, returning the new 2D image
    # Args:
    #   image_list: a list of 2D images (also known as the image)
    def use_kernel (self, image_list):
        new_image_size = (len(image_list[0]) - self.feature_map_height + 1, len(image_list[0][0]) - self.feature_map_length + 1)
        new_image = np.zeros(new_image_size)
        for y in range(new_image_size[0]):
            for x in range(new_image_size[1]):
                new_image[y][x] = np.sum(np.multiply(image_list[:,y:y+self.feature_map_height,x:x+self.feature_map_length], self.weights)) + self.bias
        return new_image

    # Returns the weight, bias, and delta errors given a current set of deltas
    # Args:
    #   input_image_shape (3-tuple) - a 3 tuple for the shape of input images (num images, image height, image length)
    #   output_image_shape (3-tuple) - a 3 tuple for the shape of the output images (same format)
    #   z-activations (3D np array) - the previous non-squashed activations
    #   deltas (3D np array) - the previous errors
    def get_errors(self, input_image_shape, output_image_shape, z_activations, deltas):
        deltaPrevs = []
        weightDeltas = []
        biasDelta = 0

        for w, z in zip(self.weights, z_activations):
            w_err, b_err, d_err = prev_errors(input_image_shape[1:],output_image_shape[1:],
                                  z,w,self.bias,deltas)
            deltaPrevs.append(d_err)
            weightDeltas.append(w_err)
            biasDelta += b_err

        return np.array(weightDeltas), np.array(biasDelta), np.array(deltaPrevs)

    # Updates the kernels weights and biases
    #   d_weight (3D np arr) - what to add to the weights
    #   d_bias (float) - what to add to the bias
    def update (self, d_weight, d_bias):
        self.weights += d_weight
        self.bias += d_bias

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias
