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
import seaborn as sns

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
class_names = train_ds.class_names

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(32):
            ax = plt.subplot(8,4,i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("images_with_labels.png",dpi=120) 
    plt.close()
  
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
epochs=1
history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs) 

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.tight_layout()
plt.savefig("training_and_validation_accuracy.png",dpi=120) 
plt.close()
  

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.tight_layout()
plt.savefig("training_and_validation_loss.png",dpi=120) 
plt.close()

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print('\nTest accuracy:', test_acc)

data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("augumented_images.png",dpi=120) 
    plt.close()

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.tight_layout()
plt.savefig("training_and_validation_accuracy2.png",dpi=120) 
plt.close()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.tight_layout()
plt.savefig("training_and_validation_loss2.png",dpi=120) 
plt.close()
test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print('\nTest accuracy:', test_acc)

glass_url = "https://sc02.alicdn.com/kf/HTB1U6akaOzxK1RkSnaVq6xn9VXac.jpg_350x350.jpg"
glass_path = tf.keras.utils.get_file('glass_bottle', origin=glass_url)

img = tf.keras.preprocessing.image.load_img(
    glass_path, target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
# Simple Bar Plot
plt.figure(figsize=(7,5))
plt.xticks(rotation=90)
plt.bar(class_names,score,align='center',width=1)
plt.xlabel('Categories')
plt.ylabel("Values")
plt.title('Predictions Bar Plot')
plt.show()
plt.tight_layout()
plt.savefig("all_predictions_probabilities.png",dpi=120) 
plt.close()
