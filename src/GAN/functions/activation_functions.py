import numpy as np

# leaky relu function
class LeakyRELU:
    # function
    @staticmethod
    def func (z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 0:
                return z
            else:
                return 0.1*z

        for i, zi in enumerate(z):
            z[i] = LeakyRELU.func(zi)
        return z

    # Derivative for leaky relu
    @staticmethod
    def func_deriv(z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 0:
                return 1
            else:
                return 0.1
        for i, zi in enumerate(z):
            z[i] = LeakyRELU.func_deriv(zi)
        return z

class Sigmoid:
    # function
    @staticmethod
    def func (z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 15:
                return 0.999999999
            elif z < -15:
                return 0.000000001
            else:
                return np.exp(z)
        for i, zi in enumerate(z):
            z[i] = Sigmoid.func(zi)
        return z

    # func derivative
    @staticmethod
    def func_deriv (z):
        if isinstance(z, float) or isinstance(z, int):
            z = Sigmoid.func(z)
            return z*(1-z)
        for i, zi in enumerate(z):
            z[i] = Sigmoid.func_deriv(zi)
        return z

# Softmax function
class Softmax:
    # used to raise powers to e
    @staticmethod
    def get_exp(z):
        if isinstance(z, float) or isinstance(z, int):
            if z > 10:
                return np.exp(10.0) + z - 9
            elif z < -15:
                return np.exp(-15.0 - np.log(-(z + 14.0)))
            else:
                return np.exp(z)
        for i, zi in enumerate(z):
            z[i] = Softmax.get_exp(zi)
        return z

    # softmax func
    @staticmethod
    def func(z):
        z = Softmax.get_exp(z)
        return z / np.sum(z)

    #derivative of softmax (z*(1-z)) for unsquashed activations z
    @staticmethod
    def func_deriv(z):
        z = Softmax.func(z)
        return z*(1-z)
