from neuralnets import ConvolutionalNet as CNN
import numpy as np
inp = [np.array([0, 0, 0]),
       np.array([0, 0, 1]),
       np.array([0, 1, 0]),
       np.array([0, 1, 1]),
       np.array([1, 0, 0]),
       np.array([1, 0, 1]),
       np.array([1, 1, 0]),
       np.array([1, 1, 1])]

outp = [np.array([1, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0, 1, 0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 1, 0, 0, 0, 0, 0]),
        np.array([0, 0, 0, 1, 0, 0, 0, 0]),
        np.array([0, 0, 0, 0, 1, 0, 0, 0]),
        np.array([0, 0, 0, 0, 0, 1, 0, 0]),
        np.array([0, 0, 0, 0, 0, 0, 1, 0]),
        np.array([0, 0, 0, 0, 0, 0, 0, 1])]

dn = CNN(3)
dn.add(layer_type="dense", output_size=20)
dn.add(layer_type="soft", output_size=8)

training_inputs = []
training_outputs = []
for x in range(1000):
    training_inputs.append(inp[x%8])
    training_outputs.append(outp[x%8])

for x in range (10000):
    print(x)
    ss = 0.05
    if x%20 == 39:
        ss *= 0.9
    dn.momentum_based_sgd(epochs=1,
                           resistance=0.9,
                           step_size=0.01,
                            mini_batch_size=80,
                            training_inputs=training_inputs,
                            expected_outputs=training_outputs)
    for c, i in enumerate(inp):
        print(dn.feedforward(i)[c])
    print ("b")
    print dn.layers[-1].biases