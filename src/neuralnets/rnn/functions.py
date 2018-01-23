import numpy as np

# Returns -1 if number is negative, 1 if positive, 0 if 0
def sign (num):
    return 1 if num > 0 else 0

# -- TanH logistic function
class TanH:
    @staticmethod
    def func (a):
        num = np.exp(a)-np.exp(-a)
        dem = np.exp(a)+np.exp(-a)
        return num/dem

    @staticmethod
    def func_deriv(a):
        return 1 - TanH.func(a)**2

# -- Class defining the sigmoid activation function
class Sigmoid:
    @staticmethod
    # Returns sigmoid function applies to a value a by the mapping SIG: a |--> 1/(1+e^-a)
    def func(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    # Returns the sigmoid prime of a value a by the mapping SIG_P: a |--> SIG(a)*(1-SIG(a))
    def func_deriv(a):
        z = Sigmoid.func(a)
        return z*(1-z)

# -- SOFTMAX output logistic function
class SoftMax:
    @staticmethod
    def func(a):
        a = np.exp(a)
        a /= sum(a)
        return a

    @staticmethod
    def func_deriv(a):
        a = np.exp(a)
        s = sum(a)
        d = np.zeros(len(a))
        for x in range(len(a)):
            d[x] = (s - a[x]) / s ** 2
        return d

class LeakyReLU:
    @staticmethod
    def func(a):
        if type(a) is np.ndarray:
            for x in range(len(a)):
                if a[x] < 0:
                    a[x] *= 0.001
        else:
            if a < 0:
                a *= .001
        return a

    @staticmethod
    def func_deriv(a):
        if type(a) is np.ndarray:
            for x in range(len(a)):
                a[x] = 1 if a[x] > 0 else 0.001
        else:
            if a < 0:
                return 0.001
            return 1
        return a

# -- RELU activation function
class ReLU:
    @staticmethod
    def func(a):
        if type(a) is np.ndarray:
            for x in range(len(a)):
                a[x] = max(0, a[x])
        else:
            if a < 0:
                a = 0
        return a

    @staticmethod
    def func_deriv(a):
        if type(a) is np.ndarray:
            for x in range(len(a)):
                a[x] = 1 if a[x] > 0 else 0
        else:
            if a < 0:
                return 0
            return 1
        return a



# -- Class defining quadratic cost
class QuadraticCost:
    # Returns the cost for given vectors output and expected by C = sum(norm_sqrd(exp-out))/2(num_tests)
    @staticmethod
    def cost(outs, expected):
        sum = 0;
        for o, e in zip(outs, expected):
            sum += (e-o)*(e-o)
        return sum/(2*len(outs))

    # Returns the cost for given network with L2 regularization
    @staticmethod
    def cost_with_L2_regularization(outs, expected, weights, lmbda, training_set_size):
        sum = QuadraticCost.cost(outs, expected)
        regularization_term = 0
        for l1 in weights:
            for l2 in l1:
                for l3 in l2:
                    regularization_term += l3*l3
        regularization_term *= lmbda/(2*training_set_size)
        return sum + regularization_term

    # Returns the error vector for the output layer by d = (a-y)*sig_p(z)
    @staticmethod
    def delta(a, y, z):
        return (a-y)*Sigmoid.func_deriv(z)

# -- Class defining cross entropy
class CrossEntropy:
    # Returns the cost for given output and expected vectors by C = sum(y*ln(a) + (1-y)*ln(1-a))/num_tests
    @staticmethod
    def cost(outs, expected):
        sum = 0
        for o, e in zip(outs, expected):
            sum += e*np.log(o) + (1-e)*np.log(1-o)
        return sum/len(outs)

    # Returns the cost with L2 regularization
    @staticmethod
    def cost_with_L2_regularization(outs, expected, weights, lmbda, training_set_size):
        sum = CrossEntropy.cost(outs, expected)
        regularization_term = 0
        for l1 in weights:
            for l2 in l1:
                for l3 in l2:
                    regularization_term += l3 * l3
        regularization_term *= lmbda / (2 * training_set_size)
        return sum + regularization_term

    # Returns the error vector for the output layer by d = a-y
    @staticmethod
    def delta(a, y, z):
        return (a-y)

# -- Cost function
class NegativeLogLikelihood:
    # Returns negative log likelihood cost for output a and expected output y
    @staticmethod
    def cost (a, y):
        for o, e in zip (a, y):
            if e == 1:
                if o == 0:
                    o += 0.0000000000000001
                return -np.log(o)

    # Returns the first error layer for the negative log likelihood function with out a and expected y
    @staticmethod
    def delta (a, y, z):
        return (a-y)