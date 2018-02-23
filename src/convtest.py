from neuralnets import ConvolutionalNet as CNN
import numpy as np
#from storenets import save

inp = [np.array([np.array([np.array([1, 1,1]),np.array([1,1,1]),np.array([1,1,1])])]),
       np.array([np.array([np.array([1,1,1]),np.array([1,1,1]),np.array([1,1,0])])]),
       np.array([np.array([np.array([1, 0, 1]), np.array([1, 1, 1]), np.array([1, 1, 0])])]),
       np.array([np.array([np.array([1, 0, 1]), np.array([1, 1, 0]), np.array([1, 1, 0])])]),
       np.array([np.array([np.array([1, 0, 1]), np.array([1, 0, 0]), np.array([1, 1, 0])])]),
       np.array([np.array([np.array([0, 0, 0]), np.array([0, 0, 1]), np.array([1, 1, 0])])])]
#inp = [np.array([np.random.randn(5,5)]) for x in range(6)]
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

# weights = np.array([np.array([1.0, -0.1]),np.array([2.0, -0.2]),np.array([3.0, -0.3])])
# biases = np.array([0,0,0])
# zs = np.array([1.0, -1.0])
# deltas = np.array([0.3, 0.2, 0.1])


# layer_type (str)- conv, deconv, dense, soft
# output_shape (int/tuple) - deconv: output image size - (img height, img length)
#                            soft/dense: int - num neurons on curr layer
#                            conv: None since it is predetermined by kernel info
# Kernel size (tuple)- (num kernels, kernel height, kernel length)
np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)

cnn = CNN(input_shape=(1, 3, 3))
cnn.add(layer_type="conv", output_size=None, kernel_size=(3, 2, 2))
#cnn.add(layer_type="conv", output_size=None, kernel_size=(3, 2, 2))
cnn.add(layer_type="deconv", output_size=(3,3), kernel_size=(1, 2, 2))
cnn.add(layer_type="deconv", output_size=(5,5), kernel_size=(3, 2, 2))
cnn.add(layer_type="dense", output_size=10)
cnn.add(layer_type="soft", output_size=6)

# cnn = CNN(input_shape=9)
# cnn.add(layer_type="dense", output_size=60)
# cnn.add(layer_type="dense", output_size=60)
# cnn.add(layer_type="soft", output_size=6)

#print cnn.feed_forward(inp[1])

for x in range (10000):
    print(x)
    ss = 0.05
    if x%20 == 39:
        ss *= 0.9
    cnn.stochastic_gradient_descent(epochs=5,
                                    step_size=0.001,
                                    mini_batch_size=80,
                                    training_inputs=training_inputs,
                                    expected_outputs=training_outputs,
                                    is_momentum_based=True,
                                    friction=0.5)
    for i in inp:
        print(cnn.feedforward(i))
    # print cnn.layers[0].kernels[0].weights[0]
    # print cnn.layers[5].weights[0][0]
    #save("test", cnn)


