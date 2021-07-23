import importlib
from os import path


class Config(object):

    def __init__(self, model):
        self.mod = importlib.import_module(model)
        self.mod.Evaluator.setup()
        self.content_weight = 0.025
        self.style_weight = 12.0
        self.total_variation_weight = 10.0
        self.num_iterations = 10

    def set_content(self, img):
        self.content = self.mod.Evaluator.preprocess_image(img)

    def set_style(self, img):
        self.style = self.mod.Evaluator.preprocess_image(img)
