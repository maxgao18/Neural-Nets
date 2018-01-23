from neuralnets import GAN
from neuralnets import Generator
from neuralnets import Discriminator
from neuralnets import ConvolutionalNet
from layers import Kernel
from layers import DenseLayer
from layers import SoftmaxLayer
from layers import DeconvLayer
from layers import ConvLayer

import os

# Converts a tuple to string by adding "%" between elements
def tuple_to_str(tup):
    strtup = ""
    for t in tup:
        strtup += str(t) + "%"
    return strtup

# Converts string to tuple by removing "%" between elements
def str_to_tuple(strn):
    lst = []
    while len(strn) > 0:
        ind = strn.index("%")
        lst.append(strn[:ind])
        strn = strn[ind+1:]
    return tuple(lst)

# Saves a network into a seperate folder
def save (filename, network):
    currdir = os.getcwd()
    if isinstance(network, GAN):
        currdir = currdir+"/"+filename+"_gan_net"
    elif isinstance(network, ConvolutionalNet):
        currdir = currdir+"/"+filename+"_cnn_net"
    else:
        return "Failed save"

    if not os.path.exists(currdir):
        os.makedirs(currdir)

    save_net(filename, network, currdir)

# Saves a network, layer, or kernel into a txt file
def save_net(filename, network, currdir):
    # If gan object
    if isinstance(network, GAN):
        filename += "_gan"
        filedir = os.path.join(currdir, filename+".txt")
        savefile = open(filedir, "w")

        # Save file names for generator and discriminator
        savefile.write(filename+"_gen.txt\n")
        savefile.write(filename+"_dis.txt\n")

        savefile.write(tuple_to_str(network.image_shape) + "\n")
        savefile.write(tuple_to_str(network.generator_input_shape) + "\n")
        savefile.write(str(network.discriminator_output_shape) + "\n")

        # Save generator and discriminator
        save_net(filename, network.get_generator(), currdir)
        save_net(filename, network.get_discriminator(), currdir)

        # Close file
        savefile.close()

    # If network object
    elif isinstance(network, (Generator, Discriminator, ConvolutionalNet)):
        filedir = None
        # Extend file name depending on type of network
        if isinstance(network, Generator):
            filename += "_gen"
        elif isinstance(network, Discriminator):
            filename += "_dis"
        elif isinstance(network, ConvolutionalNet):
            filename += "_cnn"

        filedir = os.path.join(currdir, filename + ".txt")

        savefile = open(filedir, "w")

        # Save number of layers
        savefile.write(str(network.num_layers) + "\n")

        # Save layer types and layers
        for i, lt, lyr in zip(range(network.num_layers), network.layer_types, network.layers):
            # Save layer type
            savefile.write(lt + "\n")
            newfilename = filename + "_" + lt + "_" +str(i)
            # Save file name of place where layer is stored
            savefile.write(newfilename + ".txt\n")
            save_net(newfilename, lyr, currdir)

        # Close file
        savefile.close()

    # If layer object
    elif isinstance(network, (ConvLayer, DeconvLayer, DenseLayer, SoftmaxLayer)):
        filedir = os.path.join(currdir, filename + ".txt")

        savefile = open(filedir, "w")

        if isinstance(network, (ConvLayer, DeconvLayer)):
            savefile.write(tuple_to_str(network.input_shape) + "\n")

            # If deconv, save output size
            if isinstance(network, DeconvLayer):
                savefile.write(tuple_to_str(network.output_shape) + "\n")

            # Save kernel shape
            savefile.write(tuple_to_str(network.kernel_shape) + "\n")

            # Save all kernels
            for i, k in enumerate(network.kernels):
                newfilename = filename + "_kern_" + str(i)
                # Save location of kernel text file
                savefile.write(newfilename+".txt\n")
                save_net(newfilename, k, currdir)

        elif isinstance(network, (SoftmaxLayer, DenseLayer)):
            savefile.write(tuple_to_str(network.layer_shape) + "\n")
            weights = network.weights
            biases = network.biases

            # Save weights (2D arr)
            for w1 in weights:
                for w2 in w1:
                    savefile.write(str(w2) + "\n")

            # Save biases (1D arr)
            for b1 in biases:
                savefile.write(str(b1) + "\n")

        # Close file
        savefile.close()

    # If kernel object
    elif isinstance(network, Kernel):
        filedir = os.path.join(currdir, filename + ".txt")

        savefile = open(filedir, "w")

        savefile.write(tuple_to_str(network.kernel_size) + "\n")

        weights = network.weights
        bias = network.bias

        # Save weights (3D)
        for w1 in weights:
            for w2 in w1:
                for w3 in w2:
                    savefile.write(str(w3) + "\n")

        # Save bias (float)
        savefile.write(str(bias) + "\n")

        # Close file
        savefile.close()