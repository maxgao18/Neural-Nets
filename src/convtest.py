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

# layer_type (str)- conv, deconv, dense, soft
# output_shape (int/tuple) - deconv: output image size - (img height, img length)
#                            soft/dense: int - num neurons on curr layer
#                            conv: None since it is predetermined by kernel info
# Kernel size (tuple)- (num kernels, kernel height, kernel length)


#layer_types = ["conv", "conv", "deconv", "deconv", "dense", "soft"]
#layer_shapes = [[(1,3,3),(1,1,1,1)], [(1,3,3), (3, 1, 2, 2)], [(3,2,2),(1,3,3),(1,3,2,2)], [(1,3,3), (3, 5, 5), (3, 1, 2, 2)], [(10, 3*5*5)], [(6, 10)]]
#layer_shapes = [[(3,120,120),(3,3,61,61)], [(3,60,60),(2,3,31,31)], [(2,30,30),(1,40,40),(1,2,10,10)], [(20,1*40*40)], [(5,20)]]
#cnn = CNN(layer_types, layer_shapes)
cnn = CNN(input_shape=(1, 3, 3))
cnn.add(layer_type="conv", output_size=None, kernel_size=(1, 1, 1))
cnn.add(layer_type="conv", output_size=None, kernel_size=(3, 2, 2))
cnn.add(layer_type="deconv", output_size=(3,3), kernel_size=(1, 2, 2))
cnn.add(layer_type="deconv", output_size=(5,5), kernel_size=(3, 2, 2))
cnn.add(layer_type="dense", output_size=10)
cnn.add(layer_type="soft", output_size=6)

print cnn.feed_forward(inp[1])
cnn.stochastic_gradient_descent(epochs=8,
                                step_size=0.00001,
                                mini_batch_size=80,
                                training_inputs=training_inputs,
                                expected_outputs=training_outputs)

#cnn.stochastic_gradient_descent(1,0.000001,1,[np.zeros((3,120,120))],[np.zeros(5)])
