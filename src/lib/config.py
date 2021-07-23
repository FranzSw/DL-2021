import importlib
from os import path


class Config(object):

    def __init__(self, model):
        if type(model) == str:
            model = importlib.import_module(model).Evaluator
        self.evaluator = model
        self.evaluator.setup()
        self.content_weight = 0.025
        self.style_weight = 12.0
        self.total_variation_weight = 10.0
        self.num_iterations = 10

    def set_content(self, img):
        self.content = self.evaluator.preprocess_image(img)

    def set_style(self, img):
        self.style = self.evaluator.preprocess_image(img)
