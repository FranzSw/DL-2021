import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from scipy.optimize import minimize

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
    return tf.reduce_mean(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def _total_variation_loss(x):
    a = backend.square(x[:, :Evaluator.height-1, :Evaluator.width -
                       1, :] - x[:, 1:, :Evaluator.width-1, :])
    b = backend.square(x[:, :Evaluator.height-1, :Evaluator.width -
                       1, :] - x[:, :Evaluator.height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


class Evaluator(object):
    width = 224
    height = 224

    @classmethod
    def setup(_cls):
        tf.compat.v1.disable_eager_execution()

    @classmethod
    def preprocess_image(cls, img):
        img = img.resize((cls.width, cls.height))
        img = np.asarray(img)
        img = np.asarray(img, dtype='float32')
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img

    @classmethod
    def postprocess_image(cls, img):
        img = img.reshape((cls.height, cls.width, 3))

        # Inverse of the preprocessing function to get viewable image
        img = img[:, :, ::-1]
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        img = np.clip(img, 0, 255).astype('uint8')

        return Image.fromarray(img)

    def __init__(self, config):
        # Copy necessary configuration
        self.loss_value = None
        self.grads_values = None
        self.content_weight = config.content_weight
        self.style_weight = config.style_weight
        self.total_variation_weight = config.total_variation_weight

        # Prepare variables for computation graph
        content_image = backend.variable(config.content)
        style_image = backend.variable(config.style)
        self.combination_image = backend.placeholder(
            (1, Evaluator.height, Evaluator.width, 3))
        self.input_tensor = backend.concatenate([content_image,
                                                 style_image,
                                                 self.combination_image], axis=0)

        # Setup classifier model
        self.model = ResNet50(
            input_tensor=self.input_tensor, include_top=False, weights='imagenet')
        self.layers = dict([(layer.name, layer.output)
                           for layer in self.model.layers])

        # Define loss computation graph
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
        res = minimize(self.eval_loss, self.x.flatten(),
                       method='L-BFGS-B', jac=self.eval_grads, options={'maxiter': 20})
        self.x = res.x
        return self.x

    def content_loss(self):
        layer_features = self.layers['conv1_conv']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        return self.content_weight * _content_loss(content_image_features,
                                                   combination_features)

    def style_loss(self):
        loss = backend.variable(0)
        # feature_layers = ['bn2b_branch2a', 'bn3a_branch2c',
        #                   'res3a_branch1', 'bn4e_branch2b']
        feature_layers = ['conv2_block2_1_bn', 'conv3_block1_0_bn',
                          'conv3_block1_3_conv', 'conv4_block5_2_bn']
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
