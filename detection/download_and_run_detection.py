from utils import ALL_MODELS, IMAGES_FOR_TEST, load_image_into_numpy_array, save_model
from config import MODEL_NAME
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import os
import pathlib

# import matplotlib
# import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO


import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel('ERROR')

"""
code from https://www.tensorflow.org/hub/tutorials/tf2_object_detection?hl=ko
"""

# %matplotlib inline

PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)
# @param ['Beach', 'Dogs', 'Naxos Taverna', 'Beatles', 'Phones', 'Birds']
selected_image = 'Beach'

# Apply image detector on a single image.


def load_model(MODEL_NAME):
    model_display_name = MODEL_NAME
    model_handle = ALL_MODELS[model_display_name]

    print('Selected model:' + model_display_name)
    print('Model Handle at TensorFlow Hub: {}'.format(model_handle))

    print('loading model...')
    hub_model = hub.load(model_handle)
    print('model loaded!')

    return hub_model
    # Data prepare


def prepare_data(flip_image_horizontally=False, convert_image_to_grayscale=False):
    image_path = IMAGES_FOR_TEST[selected_image]
    image_np = load_image_into_numpy_array(image_path)

    # Flip horizontally
    if(flip_image_horizontally):
        image_np[0] = np.fliplr(image_np[0]).copy()

    # Convert image to grayscale
    if(convert_image_to_grayscale):
        image_np[0] = np.tile(
            np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    return image_np


if __name__ == "__main__":
    image_np = prepare_data()
    model = load_model(MODEL_NAME)
    save_model(model, "saved_model_detection")
    results = model(image_np)

    # different object detection models have additional results
    # all of them are explained in the documentation
    result = {key: value.numpy() for key, value in results.items()}
    print(result.keys())
