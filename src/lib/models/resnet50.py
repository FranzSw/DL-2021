from keras.applications.resnet import ResNet50
from ..helpers import *
from ..evaluator import Evaluator as BaseEvaluator


class Evaluator(BaseEvaluator):
    dimensions = (224, 224)
    style_layers = ['conv2_block2_1_bn', 'conv3_block1_0_bn',
                    'conv3_block1_3_conv', 'conv4_block5_2_bn']
    content_layers = ['conv1_conv']

    @classmethod
    def preprocess_image(cls, img):
        return preprocess_image_imagenet(img, cls.dimensions)

    @classmethod
    def postprocess_image(cls, img):
        return postprocess_image_imagenet(img, cls.dimensions)

    def setup_model(self):
        self.model = ResNet50(input_tensor=self.input_tensor, weights='imagenet',
                              include_top=False)
