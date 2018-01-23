from generator import Generator
from discriminator import Discriminator

class GAN:
    # Args:
    #   image_shape (3 tuple) - the shape of the image desired to be created by the generator
    #                               and inputted into the discriminator (image depth, image height, image length)
    #   generator_input_shape (3 tuple) - the shape of the image (noise) to be fed to the generator
    #   discriminator_output_shape (int) - number of desired outputs from the discriminator
    def __init__(self, image_shape, generator_input_shape, discriminator_output_shape):
        self.image_shape = image_shape
        self.generator_input_shape = generator_input_shape
        self.discriminator_output_shape = discriminator_output_shape

        self.generator = Generator(generator_input_shape)
        self.discriminator = Discriminator(image_shape)

    # Trains the generator against the discriminator using stochastic gradient descent
    # Args:
    #   epochs (int) - number of times to loop over entire training set
    #   step_size (float)
    #   mini_batch_size (int) - number of training inputs per mini batch
    #   training set (list of tuples) - a list of tuples
    #                                   (training input (noise), desired output (fool the discriminator))
    def train_generator(self, epochs, step_size, mini_batch_size, training_set):
        self.generator.stochastic_gradient_descent(epochs,
                                                   step_size,
                                                   mini_batch_size,
                                                   training_set,
                                                   self.discriminator)

    # Trains discriminator against the discriminator against real images and the generated images
    # Args:
    #   ...
    #   training_set (list of tuples) - list of tuples,
    #                                   (generated image/real image, expected output (from discriminator))
    def train_discriminator(self, epochs, step_size, mini_batch_size, training_set):
        self.discriminator.stochastic_gradient_descent(epochs,
                                                       step_size,
                                                       mini_batch_size,
                                                       training_set)

    # Generates an image from noise using the current generator
    # Args:
    #   noise (3D np array)
    def generate_image(self, noise):
        return self.generator.feed_forward(noise)

    # Adds layer to generator
    # Args:
    #   layer_type (string) - "conv" or "deconv"
    #   output_size (tuple) - input None if conv layer, else desired output image shape
    #   kernel_size (tuple) - (num of kernels, kernel height, kernel length)
    def add_layer_to_generator(self, layer_type, output_size, kernel_size):
        self.generator.add(layer_type, output_size, kernel_size)

    # Adds layer to discriminator
    # Args:
    #   layer_type (string) - conv, deconv, soft, or dense
    #   output_size (tuple/int) - input None if conv layer, desired output image shape if dense,
    #                               number of neurons if dense/soft
    #   kernel_size (tuple) - if soft/dense, None, else (num of kernels, kernel height, kernel length)
    def add_layer_to_discriminator(self, layer_type, output_size, kernel_size=None):
        self.discriminator.add(layer_type, output_size, kernel_size)

    # Feeds image through generator
    def generator_feed_forward (self, network_input):
        return self.generator.feed_forward(network_input)

    # Feeds image through discriminator
    def discriminator_feed_forward(self, network_input):
        return self.discriminator.feed_forward(network_input)

    def get_image_shape(self):
        return self.image_shape

    def get_generator_input_shape(self):
        return self.generator_input_shape

    def get_discriminator_output_shape(self):
        return self.discriminator_output_shape

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator