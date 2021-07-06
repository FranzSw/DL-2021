
import importlib
from PIL import Image
from glob import glob
import os
from os import path
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import itertools


models = ['vgg16']
content_weights = [0.025, 0.08, 0.15]
style_weights = [2.0, 5.0, 12.0]
total_variation_weights = [1.0, 3.0, 10.0]
num_iterations = [10]
# num_iterations = range(1, 11)
# num_iterations = range(1, 11, 2)

folder, _ = os.path.split(os.path.realpath(__file__))
out_folder = path.join(folder, 'out')
content_folder = path.join(folder, 'in/content/*')
style_folder = path.join(folder, 'in/style/*')
overwrite = False


def calculate(mod, content, style, content_weight, style_weight, total_variation_weight, out_base):
    print(out_base)
    evaluator = mod.Evaluator(content, style, content_weight,
                              style_weight, total_variation_weight)
    for i in range(1, max(num_iterations)+1):
        print('Start of iteration', i)
        x = evaluator.eval_and_train()
        if i in num_iterations:
            img = x.copy()
            mod.Evaluator.postprocess_image(img).save(
                path.join(out_folder, f'{out_base}_{i}.png'))
    print()


def process(content_image, style_image, out):
    for model in models:
        mod = importlib.import_module(model)
        mod.Evaluator.setup()
        for (content_weight, style_weight, total_variation_weight) in itertools.product(content_weights, style_weights, total_variation_weights):
            content = mod.Evaluator.preprocess_image(content_image)
            style = mod.Evaluator.preprocess_image(style_image)
            out_base = f'{model}_{content_weight}_{style_weight}_{total_variation_weight}'
            out_base = path.join(out, out_base)
            if overwrite or not all([path.exists(path.join(out_folder, f'{out_base}_{i}.png'))
                                     for i in num_iterations]):
                calculate(mod, content, style, content_weight,
                          style_weight, total_variation_weight, out_base)


def load_image_rgb(file):
    img = Image.open(file)
    if img.mode != 'RGB':
        rgbimg = Image.new('RGB', img.size)
        rgbimg.paste(img)
        img = rgbimg
    return img


for content_file in glob(content_folder):
    content_image = load_image_rgb(content_file)
    for style_file in glob(style_folder):
        style_image = load_image_rgb(style_file)
        out = f'{path.basename(content_file)}_{path.basename(style_file)}'
        out_dir = path.join(out_folder, out)
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        process(content_image, style_image, out)
