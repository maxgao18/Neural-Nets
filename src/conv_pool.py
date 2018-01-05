import numpy as np
class ConvolutionalPoolingLayer:

    # Parameters:
    # image_shape: a 3-tuple (num_images, image_height, image_length)
    # filter_shape: a 4-tuple (num_filters, filter_depth, filter_height, filter_length)
    # pool_shape: a 2-tuple (pool_height, pool_length)
    def __init__(self, image_shape, filter_shape, pool_shape, pooling_method="maxpool"):
        self.image_shape = image_shape
        self.filter_shape = filter_shape
        self.pool_shape = pool_shape
        self.pooling_method = pooling_method

        # Create list of filter objects
        self.filters = []
        for i in range(filter_shape[0]):
            self.filters.append(Filter(self.filter_shape[1:]))

    # Forwards past a list of images and returns the new list of images
    def forward_pass (self, image_list):
        new_image_list = []
        for i in range(self.filter_shape[0]):
            fil = self.filters[i]
            feature_image_size, feature_image = fil.use_filter(self.image_shape[1:], image_list)
            if (self.pooling_method == "maxpool"):
                new_image_list.append(MaxPool.pool(feature_image_size, feature_image, self.pool_shape))
        return new_image_list


# Individual filter objects
class Filter:
    # Parameters:
    # filter_size: a 3-tuple (filter_depth, filter_height, filter_length)
    def __init__(self, filter_size):
        self.filter_shape = filter_size
        self.feature_map_length = filter_size[2]
        self.feature_map_height = filter_size[1]
        self.num_feature_maps = filter_size[0]
        self.weights = np.random.randn(filter_size)
        self.bias = np.random.random

    # Takes in a list of images and applies the filter specific to the object to the filter, returning the new 2D image
    # Parameters:
    # image_shape: a 3-tuple (num_images, image_height, image_length)
    # image_list: a list of 2D images
    def use_filter (self, image_shape, image_list):
        num_images = image_shape[0]
        new_image_size = (image_shape[1] - self.feature_map_height + 1, image_shape[2] - self.feature_map_length + 1)
        new_image = np.zeros(new_image_size)
        for i in range(num_images):
            new_image += self.use_feature_map(self.weights[i], image_list[i])
        for y in range(new_image_size[0]):
            for x in range(new_image_size[1]):
                new_image[y][x] += self.bias
        return new_image_size, new_image


    # This method takes in a feature map and slides it across an image,
    # returning a 2D array which is the new output image
    def use_feature_map (self, feature_map, new_image_size, image):
        new_image = np.zeros(new_image_size)
        for x in range(new_image_size[1]):
            for y in range(new_image_size[0]):
                img_piece = image[y:y+self.feature_map_height, x:x+self.feature_map_length]
                new_image_size[y][x] = np.dot(feature_map.ravel(), img_piece.ravel())
        return new_image

class MaxPool:
    # This method performs max pooling on an image and returns the new, max-pooled image
    @staticmethod
    def pool (image_size, image, pool_shape):
        image_length = image_size[1]
        image_height = image_size[0]

        pool_length = pool_shape[1]
        pool_height = pool_shape[0]

        new_image = np.zeros((image_height/pool_height, image_length/pool_length))
        for ny, iy in enumerate(range(0, image_height, pool_height)):
            for nx, ix in enumerate(range(0, image_length, pool_length)):
                new_image[ny][nx] = np.argmax(image[iy:iy+pool_height][ix:ix+pool_length])
        return new_image