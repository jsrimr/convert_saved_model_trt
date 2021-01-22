# from download_and_run import prepare_data
import time
import logging
import numpy as np

import tensorflow as tf
print("TensorFlow version: ", tf.__version__)

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
# train_generator, valid_generator = prepare_data()

# from config import BATCH_SIZE, SAVED_MODEL_DIR, FP16_SAVED_MODEL_DIR

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--BATCH_SIZE", type=int, default=32)
parser.add_argument("--SAVED_MODEL_DIR", type=str, help="Where to load model")
# parser.add_argument("--FP16_SAVED_MODEL_DIR", type=str, help="Where to save trt converted model")
args = parser.parse_args()

SAVED_MODEL_DIR = args.SAVED_MODEL_DIR
FP16_SAVED_MODEL_DIR = args.SAVED_MODEL_DIR+"_TFTRT_FP16/1"

import os
if __name__ == "__main__":
    os.system(f"rm -rf {FP16_SAVED_MODEL_DIR}")
    #old version tensorRT
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=SAVED_MODEL_DIR,
    conversion_params=conversion_params)
    converter.convert()

    converter.save(FP16_SAVED_MODEL_DIR)

    # Now we create the TFTRT FP16 engine

    # # option 1
    # tf.experimental.tensorrt.ConversionParams(precision_mode='FP16')
    # converter = tf.experimental.tensorrt.Converter(input_saved_model_dir=SAVED_MODEL_DIR)
    # # converter = trt.TrtGraphConverter(input_saved_model_dir=SAVED_MODEL_DIR,
    # #                                 max_batch_size=BATCH_SIZE,
    # #                                 precision_mode=trt.TrtPrecisionMode.FP16)
    # converter.convert()
    # converter.save(FP16_SAVED_MODEL_DIR)


    #option 2 : seems faster but hard to implement
    # params = tf.experimental.tensorrt.ConversionParams(
    #      precision_mode='FP16',
    #      # Set this to a large enough number so it can cache all the engines.
    #      maximum_cached_engines=16)
    # converter = tf.experimental.tensorrt.Converter(
    #      input_saved_model_dir=SAVED_MODEL_DIR, conversion_params=params)
    # converter.convert()

    # def my_input_fn():
    #     input_sizes = [[112, 112], [224, 224]]
    #     for size in input_sizes:
    #         inp1 = np.random.normal(size=(1, *size, 3)).astype(np.float32)
    #         yield [inp1]

    # converter.build(input_fn=my_input_fn)  # Generate corresponding TRT engines
    # converter.save(FP16_SAVED_MODEL_DIR)  # Generated engines will be saved.