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

def splitDataset(dataset, trainSize, testingSize):
    # 80% training = 52 batches
  train_dataset = dataset.take(trainSize)
    # 10% testing = 7 batches
  testing_dataset = dataset.skip(trainSize).take(testingSize)
    # 10% validation = 6 batches
  validation_dataset = dataset.skip(trainSize).skip(testingSize)