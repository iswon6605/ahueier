import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#import pandas as pd
import tensorflow as tf
#import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#tfds.disable_progress_bar()
#import math
import matplotlib.pyplot as plt
import numpy as np
import logging
import glob

#model=load_model("")

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def normalize(images, labels):
    images=tf.cast(images, tf.float32)
    images/=256
    return images,labels

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz",extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classs = ['roses', 'daisy', 'danedlion','sunflowers','tulips']

for cl in classs :
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path+'/*.jpg')
    num_train = int(round(len(images))*0.8)
    train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.2):]

    for i in train :
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))

    for i in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

BATCH_SIZE=100
IMG_SHAPE=150

image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=1)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=1,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE))

image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=train_dir,
                                                              shuffle=1,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE),)

image_gen = ImageDataGenerator(rescale=1./255, zoom_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=train_dir,
                                                              shuffle=1,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE),)

#sample_training_images, _ = next(train_data_gen)

image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=1,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE,IMG_SHAPE),
                                                 class_mode='sparse')
#plotImages(sample_training_images[:5])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

EPOCHS = 80
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(BATCH_SIZE)))
)
