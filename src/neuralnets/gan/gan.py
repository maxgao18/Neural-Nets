from generator import Generator
from discriminator import Discriminator

class GAN:
    def __init__(self, image_shape, generator_input_shape, discriminator_output_shape):
        self.image_shape = image_shape
        self.generator_input_shape = generator_input_shape
        self.discriminator_output_shape = discriminator_output_shape

        self.generator = Generator(generator_input_shape)
        self.discriminator = Discriminator(image_shape)

    def train_generator(self, epochs, step_size, mini_batch_size, training_set):
        self.generator.stochastic_gradient_descent(epochs,
                                                   step_size,
                                                   mini_batch_size,
                                                   training_set,
                                                   self.discriminator)

    def train_discriminator(self, epochs, step_size, mini_batch_size, training_set):
        self.discriminator.stochastic_gradient_descent(epochs,
                                                       step_size,
                                                       mini_batch_size,
                                                       training_set)

    def generate_image(self, image):
        return self.generator.feed_forward(image)

    def add_layer_to_generator(self, layer_type, output_size, kernel_size):
        self.generator.add(layer_type, output_size, kernel_size)

    def add_layer_to_discriminator(self, layer_type, output_size, kernel_size= None):
        self.discriminator.add(layer_type, output_size, kernel_size)

    def generator_feed_forward (self, network_input):
        return self.generator.feed_forward(network_input)

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