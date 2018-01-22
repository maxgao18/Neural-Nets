from gan import GAN
import mnist_loader
import numpy as np

gan = GAN(image_shape=(1,3,3), generator_input_shape=(1,2,2), discriminator_output_shape=2)
gan.add_layer_to_generator("deconv", (3,3), (1,2,2))
gan.add_layer_to_discriminator("conv", None, (3,2,2))
gan.add_layer_to_discriminator("dense", 10)
gan.add_layer_to_discriminator("soft", 2)

inp = [np.array([np.array([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 1])])]),
       np.array([np.array([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 1, 0])])]),
       np.array([np.array([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 0, 1])])]),
       np.array([np.array([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([1, 0, 0])])]),
       np.array([np.array([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0, 1, 1])])]),
       np.array([np.array([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0, 1, 0])])]),
       np.array([np.array([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0, 0, 1])])]),
       np.array([np.array([np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0, 0, 0])])]),
       np.array([np.array([np.array([0, 0, 1]), np.array([1, 0, 1]), np.array([1, 1, 1])])]),
       np.array([np.array([np.array([1, 1, 0]), np.array([0, 1, 0]), np.array([1, 0, 1])])])]

noise = [np.random.randn(1,2,2) for i in range(1000)]
noise_set = [(n, np.array([1,0])) for n in noise]
real_images = [(inp[i%10], np.array([1,0])) for i in range(1000)]

for x in range (100):
    generated_images = [(gan.generate_image(n), np.array([0,1])) for n in noise]

    training_set = []
    training_set.extend(real_images)
    training_set.extend(generated_images)

    print gan.generator_feed_forward(inp[0])

    gan.train_discriminator(5, 0.0001, 50, training_set)
    gan.train_generator(20, 0.0002, 50, noise_set)
