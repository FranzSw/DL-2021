import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b


def _content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def _style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = Evaluator.height * Evaluator.width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def _total_variation_loss(x):
    a = backend.square(x[:, :Evaluator.height-1, :Evaluator.width -
                       1, :] - x[:, 1:, :Evaluator.width-1, :])
    b = backend.square(x[:, :Evaluator.height-1, :Evaluator.width -
                       1, :] - x[:, :Evaluator.height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


class Evaluator(object):

    width = 512
    height = 512

    @classmethod
    def setup(_cls):
        tf.compat.v1.disable_eager_execution()

    @classmethod
    def preprocess_image(cls, img):
        img = img.resize((cls.width, cls.height))
        img = np.asarray(img)
        img = np.asarray(img, dtype='float32')
        img = np.expand_dims(img, axis=0)
        img[:, :, :, 0] -= 103.939
        img[:, :, :, 1] -= 116.779
        img[:, :, :, 2] -= 123.68
        img = img[:, :, :, ::-1]
        return img

    @classmethod
    def postprocess_image(cls, img):
        img = img.reshape((cls.height, cls.width, 3))
        img = img[:, :, ::-1]
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        img = np.clip(img, 0, 255).astype('uint8')
        img = Image.fromarray(img)
        return img

    def __init__(self, config):
        self.loss_value = None
        self.grads_values = None
        self.content_weight = config.content_weight
        self.style_weight = config.style_weight
        self.total_variation_weight = config.total_variation_weight
        content_image = backend.variable(config.content)
        style_image = backend.variable(config.style)
        self.combination_image = backend.placeholder(
            (1, Evaluator.height, Evaluator.width, 3))
        self.input_tensor = backend.concatenate([content_image,
                                                 style_image,
                                                 self.combination_image], axis=0)
        self.model = VGG16(input_tensor=self.input_tensor, weights='imagenet',
                           include_top=False)
        self.layers = dict([(layer.name, layer.output)
                           for layer in self.model.layers])
        loss = backend.variable(0)\
            + self.content_loss()\
            + self.style_loss()\
            + self.total_variation_loss()
        grads = backend.gradients(loss, self.combination_image)
        outputs = [loss]
        outputs += grads
        self.f_outputs = backend.function(
            [self.combination_image], outputs)
        self.x = np.random.uniform(
            0, 255, (1, Evaluator.height, Evaluator.width, 3)) - 128

    def eval_and_train(self):
        self.x, _min_val, _info = fmin_l_bfgs_b(self.eval_loss, self.x.flatten(),
                                                fprime=self.eval_grads, maxfun=20)
        return self.x

    def content_loss(self):
        layer_features = self.layers['block2_conv2']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        return self.content_weight * _content_loss(content_image_features,
                                                   combination_features)

    def style_loss(self):
        loss = backend.variable(0)
        feature_layers = ['block1_conv2', 'block2_conv2',
                          'block3_conv3', 'block4_conv3',
                          'block5_conv3']
        for layer_name in feature_layers:
            layer_features = self.layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = _style_loss(style_features, combination_features)
            loss = loss + (self.style_weight / len(feature_layers)) * sl

        return loss

    def total_variation_loss(self):
        return self.total_variation_weight * _total_variation_loss(self.combination_image)

    def eval_loss_and_grads(self, x):
        x = x.reshape((1, Evaluator.height, Evaluator.width, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values

    def eval_loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def eval_grads(self, _x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
