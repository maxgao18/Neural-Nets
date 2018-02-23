from abc import ABCMeta, abstractmethod
class Layer(object):
    def __init__(self, input_shape, output_shape, activation_function):
        __metaclass__ = ABCMeta
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation_function = activation_function

    def get_output_shape(self):
        return self.output_shape

    def get_input_shape(self):
        return self.input_shape

    @abstractmethod
    def feedforward(self, inputs):
        pass

    @abstractmethod
    def getactivations(self, inputs):
        pass

    @abstractmethod
    def update(self, d_weights, d_biases):
        pass

    @abstractmethod
    def backprop(self, prev_fz_activations, d_prev_z_activations, curr_deltas):
        pass

    @abstractmethod
    def getdeltas(self, d_prev_z_activations, curr_deltas):
        pass