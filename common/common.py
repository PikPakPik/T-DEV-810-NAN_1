import tensorflow as tf
import numpy as np
from IPython.display import Image, display
import matplotlib.cm as cm
from keras.utils import load_img, img_to_array

def get_img_array(img_path, size):
    # `img` is a PIL image of size 256x256
    img = load_img(img_path, target_size=size, color_mode='grayscale')
    # `array` is a float32 Numpy array of shape (256, 256, 3)
    array = img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array
