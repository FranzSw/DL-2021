
from PIL import Image, ImageEnhance
from glob import glob
from os import path
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import itertools


from config import Config
from parameters import *
from calculation import calculate
from utils import load_image_rgb


def process(content_image, style_image, out):
    for model in models:
        config = Config(model)
        for (content_weight, style_weight, total_variation_weight) in itertools.product(content_weights, style_weights, total_variation_weights):
            config.set_content(content_image)
            config.set_style(style_image)
            config.content_weight = content_weight
            config.style_weight = style_weight
            config.total_variation_weight = total_variation_weight
            config.num_iterations = max(num_iterations)
            out_base = f'{model}_{content_weight}_{style_weight}_{total_variation_weight}'
            out_base = path.join(out, out_base)
            if overwrite or not all([path.exists(path.join(out_folder, f'{out_base}_{i}_{output_saturation}.png'))
                                     for i, output_saturation in itertools.product(num_iterations, output_saturations)]):
                print(out_base)
                res = calculate(config, lambda img, i: img.save(
                    path.join(out_folder, f'{out_base}_{i}.png')) if i in num_iterations else None)
                for output_saturation in output_saturations:
                    res = ImageEnhance.Color(
                        res.copy()).enhance(output_saturation)
                    res.save(
                        path.join(out_folder, f'{out_base}_{config.num_iterations}_{output_saturation}.png'))
                print()


for content_file in glob(content_folder):
    content_image = load_image_rgb(content_file)
    for style_file in glob(style_folder):
        style_image = load_image_rgb(style_file)
        for content_saturation in content_saturations:
            content = ImageEnhance.Color(
                content_image).enhance(content_saturation)
            for style_saturation in style_saturations:
                style = ImageEnhance.Color(
                    style_image).enhance(style_saturation)
                out = f'{path.basename(content_file)}_{content_saturation}_{path.basename(style_file)}_{style_saturation}'
                out_dir = path.join(out_folder, out)
                if not path.exists(out_dir):
                    os.mkdir(out_dir)
                process(content, style, out)
