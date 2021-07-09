import os
from os import path

models = ['vgg16']
content_weights = [0.025]
style_weights = [13.0]
total_variation_weights = [4.0]
# num_iterations = [10]
# num_iterations = range(1, 11)
num_iterations = range(1, 11, 2)
content_saturations = [1.0]
style_saturations = [0.25]
output_saturations = [1.5, 2.0, 2.5, 3.0]

folder, _ = os.path.split(os.path.realpath(__file__))
out_folder = path.join(folder, 'out')
content_folder = path.join(folder, 'in/content/*')
style_folder = path.join(folder, 'in/style/*')
overwrite = False
