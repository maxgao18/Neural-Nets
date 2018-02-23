from abc import ABCMeta, abstractmethod
class NeuralNetwork(object):
    def __init__(self, network_type, cost_function, layers=None):
        __metaclass__ = ABCMeta
        self.network_type=network_type
        self.cost_function=cost_function
        self.layers=[]
        self.num_layers=0
        self.velocity=None

        if layers is not None:
            self.layers=layers
            self.num_layers=len(layers)

    def reset_velocity(self):
        self.velocity=None

    # Evaluates the average cost across the training set
    def evaluate_cost(self, training_set):
        total = 0.0
        for inp, outp in training_set:
            net_outp = self.feedforward(inp)
            total += self.cost_function.cost(net_outp, outp)
        return total/len(training_set)

    @abstractmethod
    def feedforward(self, inputs):
        pass

    @abstractmethod
    def addlayer(self, layer_type, output_size=None, kernel_size=None):
        pass

    @abstractmethod
    def stochastic_gradient_descent(self, epochs, step_size, mini_batch_size, training_inputs, expected_outputs,
                                    is_momentum_based=False, friction=0.9):
        pass

