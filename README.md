# Neural-Nets

**Project**  
This project contains different neural networks I have worked on, written in Python.

The file *functions.py* contains the non-linear functions such as sigmoid and tanh as well as various cost functions like cross-entropy.

As to date, I have created a feedforward (fully dense) neural network *feedforward.py*, a recurrent neural network *recurrent.py*, and a convolutional neural network *convolutional.py* without the use of machine learning libraries such as Tensorflow. The feedforward neural network also has L1 and L2 regularization implemented.

Currently, the feedforward, recurrent, and convolutional neural networks offer feedforward and back propagation.

**Files used for CNN**  
The files *kernel.py, dense_layer.py, conv_layer.py* are currently used by the convolutional neural network, which uses a leaky relu activation function. If you are interested in looking at my code, I have provided comments explaining the variables used in many functions and the purpose of each function in the individual python files.

I have added deconvolutional layers, under *deconv_layer.py*, which can also be used by a convolutional neural network if wanted. Back propagation for these layers has also been implemented.

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
