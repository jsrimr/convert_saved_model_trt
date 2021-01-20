import time
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.saved_model import tag_constants

N_RUN = 100

def benchmark_saved_model(SAVED_MODEL_DIR):

    saved_model_loaded = tf.saved_model.load(SAVED_MODEL_DIR, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    print('Benchmarking inference engine...')

    
    x = tf.random.uniform([1,224,224,3])
    prediction_scores = infer(x) # 초기화에 시간 오래 걸리는듯
    start_time = time.time()
    for i in range(50):
        prediction_scores = infer(x)

    # from download_and_run import build_model
    # model = build_model(5)
    # start_time = time.time()
    # for i in range(50):
    #     pred = model.predict(x)
    
       
    print(f"speed = {time.time()-start_time}")
    

if __name__ == "__main__":
    benchmark_saved_model(SAVED_MODEL_DIR="saved_model_TFTRT_FP16/1")