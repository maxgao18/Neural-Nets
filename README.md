# Neural-Nets

**Project**  
This project contains different neural networks I have worked on, written in Python.

The file *functions.py* contains the non-linear functions such as sigmoid and tanh as well as various cost functions like cross-entropy.

As to date, I have created a fully connected (fully dense) neural network *fullyconnected.py*, a recurrent neural network *recurrent.py*, and a convolutional neural network *convolutional.py* without the use of machine learning libraries such as Tensorflow. The feedforward neural network also has L1 and L2 regularization implemented.  

Also, I have included a GAN which was also coded from scratch (using a deconvolution generator network and a convolutional discriminator network).  

Currently, the feedforward, recurrent, and convolutional neural networks offer feedforward and back propagation.

**Python Packages**  
layers - includes different types of layers used  
functions - includes different activation and cost functions used  
neuralnets - includes fullyconnected, recurrent, convolutional neural network with back propagation and a gan with back propagation  

**Files used for CNN**  
The files *kernel.py, dense_layer.py, softmax_layer.py, conv_layer.py* are currently used by the convolutional neural network, as well as the gan, which uses a leaky relu or softmax activation function. If you are interested in looking at my code, I have provided comments explaining the variables used in many functions and the purpose of each function in the individual python files.

I have added deconvolutional layers, under *deconv_layer.py*, which can also be used by a convolutional neural network if wanted. Back propagation for these layers has also been implemented.

The cost functions are under *cost_functions.py* and include Quadratic Cost and Negative Log Likelihood.

The activation functions are under *activation_functions.py* and include Leaky ReLU and Softmax.  

**Initializing the CNN and adding layers to the CNN**  
*Initialization:*   
Upon initialization, the CNN object (ConvolutionalNet from *convolutional.py*) requires a single parameter, an input image size (which is a tuple of integers). The tuple should be in the order of: (input image depth, input image height, input image length). It is possible to create a CNN just as a fully connected network by allowing the input image size to be an integer, representing the number of inputs.  
  
*Adding Layers after Initialization:*  
Using the *add* method, it is possible to dynamically add layers to the CNN after its initialization. The add method takes up to 3 parameters, *layer_type (string), output_size(tuple or int), kernel_size (tuple)*.  

The layer type is a string, either "conv", "deconv", "dense", or "soft" depending on the type of layer to be added. The output size is a tuple for deconvolutional layers, which is in the order (output image height, output image length). For dense or softmax layers, a single integer representing the number of neurons to be used on that layer is to be used. Because the output shape for conv layers depends on the input image shape and the kernel shape, conv layers do not require an inputted output size (so using the *None* keyword will work). The kernel shape is used only for conv and deconv layers. It is a 3-tuple of integers, (number of kernels to be used, kernel height, kernel length).

An example can be found under *convtest.py*. This also includes an example of training the network, using stochastic gradient descent.    

**Current Goals**  
Currently, I am working on unsupervised learning and learning how to use different machine learning libraries, such as Tensorflow and Keras.

My ultimate goal is to learn how different types of neural networks work and the math behind them, and how to improve on current models.

**Main Libraries**  
numpy

**Acknowledgements:**   
http://neuralnetworksanddeeplearning.com/  
http://deeplearning.net/  
https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/  
http://mccormickml.com/2014/06/13/deep-learning-tutorial-softmax-regression/  
https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/  
