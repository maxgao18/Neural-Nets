from convolutional import Convolutional as CNN
import numpy as np
inp = [np.array([np.array([np.array([1,1,1]),np.array([1,1,1]),np.array([1,1,1])])]),
       np.array([np.array([np.array([1,1,1]),np.array([1,1,1]),np.array([1,1,0])])]),
       np.array([np.array([np.array([1, 0, 1]), np.array([1, 1, 1]), np.array([1, 1, 0])])]),
       np.array([np.array([np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([1, 1, 0])])]),
       np.array([np.array([np.array([1, 0, 1]), np.array([1, 0, 0]), np.array([1, 1, 0])])]),
       np.array([np.array([np.array([0, 0, 0]), np.array([0, 0, 1]), np.array([1, 1, 0])])])]
outp = [np.array([0,0,0,0,0,1]),
        np.array([0,0,0,0,1,0]),
        np.array([0,0,0,1,0,0]),
        np.array([0,0,1,0,0,0]),
        np.array([0,1,0,0,0,0]),
        np.array([1,0,0,0,0,0])]

training_inputs = []
training_outputs = []
for x in range(1000):
    training_inputs.append(inp[x%6])
    training_outputs.append(outp[x%6])

layer_types = ["conv", "conv", "deconv", "dense", "dense"]
layer_shapes = [[(1,3,3),(1,1,1,1)], [(1, 3, 3), (3, 1, 2, 2)], [(3,2,2),(1,3,3),(1,3,2,2)], [(10, 1*3*3)], [(6, 10)]]
cnn = CNN(layer_types, layer_shapes)
cnn.stochastic_gradient_descent(epochs=300,
                                step_size=0.0001,
                                mini_batch_size=80,
                                training_inputs=training_inputs,
                                expected_outputs=training_outputs)
