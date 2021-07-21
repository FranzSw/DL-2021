import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend
from scipy.optimize import minimize
from lib.helpers import *


class Evaluator(object):
    dimensions = None
    style_layers = None
    content_layers = None

    @classmethod
    def setup(_cls):
        tf.compat.v1.disable_eager_execution()

    @classmethod
    def preprocess_image(cls, img):
        raise NotImplementedError("Please Implement this method")

    @classmethod
    def postprocess_image(cls, img):
        raise NotImplementedError("Please Implement this method")

    def __init__(self, config):
        self.loss_value = None
        self.grads_values = None
        self.content_weight = config.content_weight
        self.style_weight = config.style_weight
        self.total_variation_weight = config.total_variation_weight
        self.content = config.content
        self.style = config.style

        self.setup_input_tensor()
        self.setup_model()
        self.setup_layers()
        self.setup_computation_graph()
        self.setup_input()

    def setup_model(self):
        self.model = None
        raise NotImplementedError("Please Implement this method")

    def setup_layers(self):
        self.layers = dict([(layer.name, layer.output)
                           for layer in self.model.layers])

    def setup_input_tensor(self):
        content_image = backend.variable(self.content)
        style_image = backend.variable(self.style)
        self.combination_image = backend.placeholder(
            (1, self.dimensions[0], self.dimensions[1], 3))
        self.input_tensor = backend.concatenate([content_image,
                                                 style_image,
                                                 self.combination_image], axis=0)

    def setup_computation_graph(self):
        loss = backend.variable(0)\
            + self.content_loss()\
            + self.style_loss()\
            + self.total_variation_loss()
        grads = backend.gradients(loss, self.combination_image)
        outputs = [loss] + grads
        self.f_outputs = backend.function(
            [self.combination_image], outputs)

    def setup_input(self):
        self.x = np.random.uniform(
            0, 255, (1, self.dimensions[0], self.dimensions[1], 3)) - 128

    def eval_and_train(self):
        res = minimize(self.eval_loss, self.x.flatten(),
                       method='L-BFGS-B', jac=self.eval_grads, options={'maxiter': 20})
        self.x = res.x
        return self.x

    def content_loss(self):
        loss = backend.variable(0)
        for layer_name in self.content_layers:
            layer_features = self.layers[layer_name]
            content_features = layer_features[0, :, :, :]
            combination_features = layer_features[2, :, :, :]
            cl = content_loss(content_features, combination_features)
            loss = loss + (self.content_weight / len(self.content_layers)) * cl

        return loss

    def style_loss(self):
        loss = backend.variable(0)
        for layer_name in self.style_layers:
            layer_features = self.layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(
                style_features, combination_features, self.dimensions)
            loss = loss + (self.style_weight / len(self.style_layers)) * sl

        return loss

    def total_variation_loss(self):
        return self.total_variation_weight * total_variation_loss(self.combination_image, self.dimensions)

    def eval_loss_and_grads(self, x):
        x = x.reshape((1, self.dimensions[0], self.dimensions[1], 3))
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
