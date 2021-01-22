import itertools
import os

# import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

module_selection = ("efficientnet", 224)
handle_base, pixels = module_selection

MODULE_HANDLE ="https://tfhub.dev/google/{}/b0/classification/1".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

from config import BATCH_SIZE

def prepare_data():
    data_dir = tf.keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)

    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                    interpolation="bilinear")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(
        data_dir, subset="validation", shuffle=True, **dataflow_kwargs)

    do_data_augmentation = False
    if do_data_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2,
            **datagen_kwargs)
    else:
        train_datagen = valid_datagen
        train_generator = train_datagen.flow_from_directory(
            data_dir, subset="training", shuffle=True, **dataflow_kwargs)

    return train_generator, valid_generator

def build_model(n_classes, do_fine_tuning = False):
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
        hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(n_classes,
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+IMAGE_SIZE+(3,))
    # model.summary()

    model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=['accuracy'])

    return model

def train(model, train_generator, valid_generator, do_fine_tuning = False):
    print("Building model with", MODULE_HANDLE)
    

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    hist = model.fit(
        train_generator,
        epochs=5, steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,
        validation_steps=validation_steps).history

from utils import save_model

def get_class_string_from_index(index):
   for class_string, class_index in valid_generator.class_indices.items():
      if class_index == index:
         return class_string

def inference(model, valid_generator):
    x, y = next(valid_generator)
    image = x[0, :, :, :]
    true_index = np.argmax(y[0])
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    # Expand the validation image to (1, 224, 224, 3) before predicting the label
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("True label: " + get_class_string_from_index(true_index))
    print("Predicted label: " + get_class_string_from_index(predicted_index))

if __name__ == "__main__":
    train_generator, valid_generator = prepare_data()
    model = build_model(n_classes=train_generator.num_classes)
    # train(model, train_generator, valid_generator)
    save_model(model)
    inference(model, valid_generator)
