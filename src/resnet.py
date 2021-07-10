import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend
from keras.models import Model
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input
from scipy.optimize import fmin_l_bfgs_b

# TODO: Same as VGG


def _content_loss(content, combination):
    return tf.reduce_mean(backend.square(combination - content))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def _style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = Evaluator.height * Evaluator.width
    return tf.reduce_mean(backend.square(S - C))


class Evaluator(object):
    width = 224
    height = 224

    @classmethod
    def setup(_cls):
        print('Setup complete')

    @classmethod
    def preprocess_image(cls, img):
        img = preprocess_input(img)
        return img

    @classmethod
    def postprocess_image(cls, img):
        # Inverse of the preprocessing function to get viewable image
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype('uint8')
        return Image.fromarray(img)

    def __init__(self, content, style, content_weight, style_weight, total_variation_weight):
        self.loss_value = None
        self.grads_value = None
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight

        content_image = backend.variable(content)
        style_image = backend.variable(style)

        self.model = ResNet50V2(include_top=False, weights='imagenet')
        self.layers = dict([(layer.name, layer.output)
                           for layer in self.model.layers])

    def content_loss(self):
        layer_features = self.layers['conv1']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        return self.content_weight * _content_loss(content_image_features,
                                                   combination_features)

    def style_loss(self):
        feature_layers = ['bn2b_branch2a', 'bn3a_branch2c',
                          'res3a_branch1', 'bn4e_branch2b']
        for layer_name in feature_layers:
            layer_features = self.layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = _style_loss(style_features, combination_features)
            loss = loss + (self.style_weight / len(feature_layers)) * sl
