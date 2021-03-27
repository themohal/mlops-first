import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import shutil

train_dir = pathlib.Path('./train')
image_count = len(list(train_dir.glob('*/*.jpg')))
print(image_count)
fileList = train_dir.glob("*/*.png")
for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

# print(listdir)
try:
    shutil.rmtree('./train/toothbrush/_taskdata')
    shutil.rmtree('./train/toothpastetube/_taskdata')
    shutil.rmtree('./train/cans/_taskdata')
    shutil.rmtree('./train/battery/_taskdata')
except OSError as error:
    print("There was an error.")

val_dir = pathlib.Path('./test')
image_count = len(list(val_dir.glob('*/*.jpg')))
print(image_count)
fileList = val_dir.glob("*/*.png")
for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

# print(listdir)
try:
    shutil.rmtree('./test/toothbrush/_taskdata')
    shutil.rmtree('./test/toothpastetube/_taskdata')
    shutil.rmtree('./test/cans/_taskdata')
    shutil.rmtree('./test/battery/_taskdata')
except OSError as error:
    print("There was an error.")
image_count = len(list(val_dir.glob('*/*.png')))
print(image_count)
# builder = tf.data.Dataset('./train')
# print(builder.info)

# trains_ds = builder.as_dataset(split='train',batch_size=32, shuffle_files=True,as_supervised=True)
# print(train_ds)
# test_ds =  builder.as_dataset(split='test',batch_size=32, shuffle_files=True,as_supervised=True)
# print(builder.info.features["label"].names)
# numpy_ds = tfds.as_numpy(trains_ds)
# numpy_ds.
# numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

num_classes = len(train_ds.class_names)

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
model.fit(train_ds,validation_data=val_ds,epochs=3)
