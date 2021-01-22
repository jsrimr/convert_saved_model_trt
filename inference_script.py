import time
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.saved_model import tag_constants

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--N_RUN", type=int, default=100)
parser.add_argument("--BATCH_SIZE", type=int, default=32)
parser.add_argument("--SAVED_MODEL_DIR", type=str, help="Where to load model ex) saved_model_TFTRT_FP16/1")
args = parser.parse_args()

SAVED_MODEL_DIR = args.SAVED_MODEL_DIR
FP16_SAVED_MODEL_DIR = args.SAVED_MODEL_DIR+"_TFTRT_FP16/1"
N_RUN = args.N_RUN

def benchmark_saved_model(SAVED_MODEL_DIR):

    saved_model_loaded = tf.saved_model.load(SAVED_MODEL_DIR, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    print('Benchmarking inference engine...')

    x = tf.random.uniform([1,1600,900,3])
    x = tf.cast(x,tf.uint8)
    prediction_scores = infer(x) # 초기화에 시간 오래 걸리는듯
    start_time = time.time()
    for i in range(N_RUN):
        prediction_scores = infer(x)

    print(f"speed = {(time.time()-start_time)/N_RUN}")
    

if __name__ == "__main__":
    benchmark_saved_model(SAVED_MODEL_DIR)