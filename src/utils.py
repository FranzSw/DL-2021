from PIL import Image


def load_image_rgb(file):
    img = Image.open(file)
    if img.mode != 'RGB':
        rgbimg = Image.new('RGB', img.size)
        rgbimg.paste(img)
        img = rgbimg
    return img
