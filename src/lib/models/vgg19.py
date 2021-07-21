from keras.applications.vgg19 import VGG19
from ..helpers import *
from ..evaluator import Evaluator as BaseEvaluator


class Evaluator(BaseEvaluator):
    dimensions = (512, 512)
    style_layers = ['block1_conv2', 'block2_conv2',
                    'block3_conv3', 'block4_conv3',
                    'block5_conv3']
    content_layers = ['block2_conv2']

    @classmethod
    def preprocess_image(cls, img):
        return preprocess_image_imagenet(img, cls.dimensions)

    @classmethod
    def postprocess_image(cls, img):
        return postprocess_image_imagenet(img, cls.dimensions)

    def setup_model(self):
        self.model = VGG19(input_tensor=self.input_tensor, weights='imagenet',
                           include_top=False)
