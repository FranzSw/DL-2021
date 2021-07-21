from keras import backend
import numpy as np
from PIL import Image


def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def style_loss(style, combination, dimensions):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = dimensions[0] * dimensions[1]
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x, dimensions):
    a = backend.square(x[:, :dimensions[1]-1, :dimensions[0] -
                       1, :] - x[:, 1:, :dimensions[0]-1, :])
    b = backend.square(x[:, :dimensions[1]-1, :dimensions[0] -
                       1, :] - x[:, :dimensions[1]-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


def preprocess_image_imagenet(img, dimensions):
    img = img.resize(dimensions)
    img = np.asarray(img)
    img = np.asarray(img, dtype='float32')
    img = np.expand_dims(img, axis=0)
    img[:, :, :, 0] -= 103.939
    img[:, :, :, 1] -= 116.779
    img[:, :, :, 2] -= 123.68
    img = img[:, :, :, ::-1]
    return img


def postprocess_image_imagenet(img, dimensions):
    img = img.reshape((dimensions[0], dimensions[1], 3))
    img = img[:, :, ::-1]
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = np.clip(img, 0, 255).astype('uint8')
    img = Image.fromarray(img)
    return img
