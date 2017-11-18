import numpy as np
import functions as fnc

class RecurrentNet:
    def __init__(self, layer_sizes, cost = fnc.NegativeLogLikelihood, logisticFunc = fnc.TanH):
        self.__n_layers = len(layer_sizes)
        self.__layer_sizes = layer_sizes
        self.__biases = [np.random.randn(s) for s in layer_sizes[1:]]
        self.__weights = [np.random.randn(cl, pl)/np.sqrt(cl)
                          for pl, cl in zip(layer_sizes[:self.__n_layers-1], layer_sizes[1:])]
        # Weights connecting current hidden layers to previous hidden layers
        self.__weights_past = [np.random.randn(cl, cl)/np.sqrt(cl) for cl in layer_sizes[1:self.__n_layers-1]]
        self.__past_hidden_state = [np.zeros(cl) for cl in layer_sizes[1:self.__n_layers-1]]
        self.__logistic_func = logisticFunc
        self.__cost = cost

    # Gets output of network
    def forward_pass(self, data):
        # Calculate next hidden state by tanh(weights*activ + weights_for_prev*activ_from_prev + biases) for hidden layers
        curr_hs = []
        for x in range(self.__n_layers-2):
            data = np.dot(self.__weights[x], data) + self.__biases[x] + np.dot(self.__weights_past[x], self.__past_hidden_state[x])
            data = self.__logistic_func.func(data)
            curr_hs.append(np.copy(data))
        data = np.dot(self.__weights[self.__n_layers-2], data) + self.__biases[self.__n_layers-2]
        data = fnc.SoftMax.func(data)

        # Update past hidden state
        self.__past_hidden_state = curr_hs
        return data

    # Calculates gradients for individual test cases
    def __back_prop (self, case, exp):
        z = case
        activations = [case]
        activations_prime = [np.zeros(self.__layer_sizes[0])]

        for w, b, pw, ps in zip(self.__weights[:self.__n_layers-2], self.__biases[:self.__n_layers-2],
                                self.__weights_past, self.__past_hidden_state):
            z = np.dot(w, z) + np.dot(pw, ps) + b
            activations.append(self.__logistic_func.func(z))
            activations_prime.append(self.__logistic_func.func_deriv(z))
            z = self.__logistic_func.func(z)
        z = np.dot(self.__weights[self.__n_layers-2], z) + self.__biases[self.__n_layers-2]
        activations.append(fnc.SoftMax.func(z))
        activations_prime.append(fnc.SoftMax.func_deriv(z))

        delta = self.__cost.delta(activations[self.__n_layers-1], exp, z)

        grad_b = [np.zeros(n) for n in self.__layer_sizes[1:]]
        grad_w = [np.zeros((c, p)) for c, p in zip(self.__layer_sizes[1:], self.__layer_sizes[:self.__n_layers-1])]
        grad_w_p = [np.zeros((c, c)) for c in self.__layer_sizes[1:self.__n_layers-1]]

        grad_b[self.__n_layers-2] = delta
        grad_w[self.__n_layers-2] = np.dot(np.array([delta]).transpose(), np.array([activations[self.__n_layers-2]]))

        for x in reversed(range(1, self.__n_layers-1)):
            delta = np.dot(self.__weights[x].T, delta)*activations_prime[x]
            grad_b[x-1] = delta
            grad_w[x-1] = np.dot(np.array([delta]).transpose(), [activations[x-1]])
            grad_w_p[x-1] = np.dot(np.array([delta]).transpose(), [self.__past_hidden_state[x-1]])

        loss = self.__cost.cost(activations[self.__n_layers-1], exp)
        return grad_b, grad_w, grad_w_p, activations, loss

    # Updates networks weights and biases based on gradients
    def __update_mini_batch (self, mini_batch, step_size):
        # Stores gradients
        net_loss = 0
        grad_b = [np.zeros(b.shape) for b in self.__biases]
        grad_w = [np.zeros(w.shape) for w in self.__weights]
        grad_w_p = [np.zeros(wp.shape) for wp in self.__weights_past]

        for data, exp in mini_batch:
            d_grad_b, d_grad_w, d_grad_w_p, act, loss = self.__back_prop(data, exp)
            grad_b = [gb+dbg for gb, dbg in zip(grad_b, d_grad_b)]
            grad_w = [gw+dgw for gw, dgw in zip(grad_w, d_grad_w)]
            grad_w_p = [gwp+dgwp for gwp, dgwp in zip(grad_w_p, d_grad_w_p)]
            net_loss += loss
            self.__past_hidden_state = act[1:self.__n_layers-1]

        change = (step_size+0.0)/len(mini_batch)
        self.__weights = [w-change*gw for w, gw in zip(self.__weights, grad_w)]
        self.__biases = [b-change*gb for b, gb in zip(self.__biases, grad_b)]
        self.__weights_past = [wp-change*gwp for wp, gwp in zip(self.__weights_past, grad_w_p)]

        return net_loss

    # Performs SDG with training inputs and outputs
    def sgd (self, training_inputs, expected_out, epochs, mini_batch_size, step_size):
        n_data = len(training_inputs)
        training_data = []
        for x in range(n_data):
            training_data.append([training_inputs[x], expected_out[x]])

        for x in range(epochs):
            loss = 0
            mini_batches = []
            st = np.random.randint(0, n_data%mini_batch_size)
            for n in range(st, n_data-mini_batch_size+st, mini_batch_size):
                mini_batches.append(training_data[n:n+mini_batch_size])
            for mini_batch in mini_batches:
                loss += self.__update_mini_batch(mini_batch, step_size)

            self.clear_mem()

            print ("epoch ", x, " loss ", loss)

    def set_weights_biases(self, weights, biases, weights_past):
        if not len(weights) == len(biases):
            return "failed"
        n_layers = len(weights) + 1
        layer_sizes = []
        layer_sizes.append(len(weights[0][0]))
        for w in weights[0]:
            if not len(w) == layer_sizes[0]:
                return "failed"
        for l, b, n in zip(weights, biases, range(0, n_layers - 1)):
            if len(l) == len(b):
                layer_sizes.append(len(b))
                for w in l:
                    if not len(w) == layer_sizes[n]:
                        return "failed"
            else:
                return "failed"

        if not len(weights_past) == n_layers-2:
            return "failed"
        for c, l in enumerate(weights_past):
            if not len(l) == layer_sizes[c+1]:
                return "failed"
            for n in l:
                if not len(n) == layer_sizes[c+1]:
                    return "failed"

        self.__n_layers = n_layers
        self.__weights = weights
        self.__biases = biases
        self.__weights_past = weights_past
        self.__layer_sizes = layer_sizes
        self.__past_hidden_state = [np.zeros(cl) for cl in layer_sizes[1:self.__n_layers-1]]
        return layer_sizes

    # Clears networks past memory
    def clear_mem(self):
        self.__past_hidden_state = [np.zeros(cl) for cl in self.__layer_sizes[1:self.__n_layers - 1]]

    def get_weights(self):
        return self.__weights

    def get_weights_past(self):
        return self.__weights_past

    def get_biases(self):
        return self.__biases

    def get_layer_sizes(self):
        return self.__layer_sizes

def write_to_file (net, filename):
    file = open(filename, 'w')
    l_s = net.get_layer_sizes()
    file.write(str(len(l_s)))
    for x in l_s:
        file.write("\n" + str(x))
    w = net.get_weights()
    for a in w:
        for b in a:
            for c in b:
                file.write("\n" + str(c))

    b = net.get_biases()
    for a in b:
        for c in a:
            file.write("\n" + str(c))

    wp = net.get_weights_past()
    for a in wp:
        for b in a:
            for c in b:
                file.write("\n" + str(c))

    file.close()
    return "wrote to " + filename

def read_w_b_from_file (filename):
    counter = 0
    file = open(filename, 'r')
    #read line by line
    arr = file.read().splitlines()
    n_layers = int(arr[counter])
    layer_sizes = []
    for x in range(n_layers):
        counter += 1
        layer_sizes.append(int(arr[counter]))
    biases = []
    weights = []
    weights_past = []
    for x in range(1, n_layers):
        layer = np.zeros((layer_sizes[x], layer_sizes[x-1]))
        for y in range(layer_sizes[x]):
            for z in range(layer_sizes[x-1]):
                counter += 1
                layer[y][z] = float(arr[counter])
        weights.append(layer)
    for x in range (1, n_layers):
        layer = np.zeros(layer_sizes[x])
        for y in range(layer_sizes[x]):
            counter += 1
            layer[x] = float(arr[counter])
        biases.append(layer)
    for x in range(1, n_layers-1):
        layer = np.zeros((layer_sizes[x], layer_sizes[x]))
        for y in range(layer_sizes[x]):
            for z in range(layer_sizes[x]):
                counter+=1
                layer[y][z] = float(arr[counter])
        weights_past.append(layer)
    file.close()

    return weights, biases, weights_past

def find_max_index (arr):
    max = arr[0]
    ind = 0
    for cnt, x in enumerate(arr[1:]):
        if max < x:
            max = x
            ind = cnt
    return ind

def to_one_hot (ind, len):
    arr = np.zeros(len)
    arr[ind] = 1
    return arr

def rand_str (net, t_len, char_to_int, int_to_char, start = None):
    num_unique_chars = len(int_to_char)
    c_len = 0
    if not start:
        ind = np.random.randint(0, t_len-1)
        start = int_to_char[ind]
        c_len = 1
    else:
        for x in range(len(start)-1):
            arr = to_one_hot(char_to_int[start[x]], num_unique_chars)
            net.forward_pass(arr)
        c_len = len(start)-1

    for x in range(c_len, t_len):
        ind = char_to_int[start[x-1]]
        nxt = find_max_index(net.forward_pass(to_one_hot(ind, num_unique_chars)))
        start += int_to_char[nxt]

    return start

'''
data = open('Nasdaq.txt', 'r').read();
print (float('123,456'))
print data
unique_chars = list(set(data))
data_size = len(data)
num_unique_chars = len(unique_chars)
print ("size, %d, unique, %d" % (data_size, num_unique_chars))

char_to_int = { ch:i for i, ch in enumerate(unique_chars)}
int_to_char = { i:ch for i, ch in enumerate(unique_chars)}

test = []
exp = []
for x in range(data_size-1):
    i1 = char_to_int[data[x]]
    test.append(to_one_hot(i1, num_unique_chars))
    i2 = char_to_int[data[x+1]]
    exp.append(to_one_hot(i2, num_unique_chars))

#w, b, wp = read_w_b_from_file("RNNtest.txt")

net = RecurrentNet([num_unique_chars, 20, 20, 20, num_unique_chars])

#print(net.set_weights_biases(w, b, wp))

#net.sgd(test, exp, 10, 200, .0001)

#write_to_file(net, "RNNtest.txt")
print (rand_str(net, 100, char_to_int, int_to_char, "big shaq"))
print (rand_str(net, 100, char_to_int, int_to_char, "bruv"))'''