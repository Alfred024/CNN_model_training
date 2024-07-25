# LIBRARIES IMPORTATION

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# DATA LOADING

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Imgs_Folder",
    shuffle = True,
    image_size= IMAGE_SIZE,
    batch_size= BATCH_SIZE # batches are like a group of various images
)

# Result = 65 batches

# SPLIT OF DATA

def splitDataset(dataset, train_bacthSize, testing_batchSize):
    # 80% training = 52 batches
  train_dataset = dataset.take(train_bacthSize)
    # 10% testing = 7 batches
  testing_dataset = dataset.skip(train_bacthSize).take(testing_batchSize)
    # 10% validation = 6 batches
  validation_dataset = dataset.skip(train_bacthSize).skip(testing_batchSize)

  return train_dataset, testing_dataset, validation_dataset
  
train_dataset, testing_dataset, validation_dataset = splitDataset(dataset, 52, 7)

# PREFETCHING + CACHING IMAGE 
# (In this case, cache would help storing 
# image data to do the processing of next epoch faster)

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
testing_dataset = testing_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# DATA AUGMENTATION 
# To make our modelo more robust 

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2)
])