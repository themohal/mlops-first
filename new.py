{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3.8.8 64-bit ('tensorflow': conda)",
   "metadata": {
    "interpreter": {
     "hash": "fc3884d88c2ef25d98786a006cab6634c00cd505ec7a26bad1c3cdcc90271f44"
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "import PIL\n",
    "import PIL.Image\n",
    "import tensorflow as tf\n",
    "import tensorflow_datasets as tfds\n",
    "import pathlib\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras.models import Sequential\n",
    "import matplotlib.pyplot as plt\n",
    "import shutil\n",
    "\n",
    "train_dir = pathlib.Path('./train')\n",
    "image_count = len(list(train_dir.glob('*/*.jpg')))\n",
    "print(image_count)\n",
    "fileList = train_dir.glob(\"*/*.png\")\n",
    "for filePath in fileList:\n",
    "    try:\n",
    "        os.remove(filePath)\n",
    "    except:\n",
    "        print(\"Error while deleting file : \", filePath)\n",
    "\n",
    "# print(listdir)\n",
    "try:\n",
    "    shutil.rmtree('./train/toothbrush/_taskdata')\n",
    "    shutil.rmtree('./train/toothpastetube/_taskdata')\n",
    "    shutil.rmtree('./train/cans/_taskdata')\n",
    "    shutil.rmtree('./train/battery/_taskdata')\n",
    "except OSError as error:\n",
    "    print(\"There was an error.\")\n",
    "\n",
    "val_dir = pathlib.Path('./test')\n",
    "image_count = len(list(val_dir.glob('*/*.jpg')))\n",
    "print(image_count)\n",
    "fileList = val_dir.glob(\"*/*.png\")\n",
    "for filePath in fileList:\n",
    "    try:\n",
    "        os.remove(filePath)\n",
    "    except:\n",
    "        print(\"Error while deleting file : \", filePath)\n",
    "\n",
    "# print(listdir)\n",
    "try:\n",
    "    shutil.rmtree('./test/toothbrush/_taskdata')\n",
    "    shutil.rmtree('./test/toothpastetube/_taskdata')\n",
    "    shutil.rmtree('./test/cans/_taskdata')\n",
    "    shutil.rmtree('./test/battery/_taskdata')\n",
    "except OSError as error:\n",
    "    print(\"There was an error.\")\n",
    "image_count = len(list(val_dir.glob('*/*.png')))\n",
    "print(image_count)\n",
    "# builder = tf.data.Dataset('./train')\n",
    "# print(builder.info)\n",
    "\n",
    "# trains_ds = builder.as_dataset(split='train',batch_size=32, shuffle_files=True,as_supervised=True)\n",
    "# print(train_ds)\n",
    "# test_ds =  builder.as_dataset(split='test',batch_size=32, shuffle_files=True,as_supervised=True)\n",
    "# print(builder.info.features[\"label\"].names)\n",
    "# numpy_ds = tfds.as_numpy(trains_ds)\n",
    "# numpy_ds.\n",
    "# numpy_images, numpy_labels = numpy_ds[\"image\"], numpy_ds[\"label\"]\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 32\n",
    "img_height = 180\n",
    "img_width = 180\n",
    "\n",
    "train_ds = tf.keras.preprocessing.image_dataset_from_directory(\n",
    "  train_dir,\n",
    "  validation_split=0.2,\n",
    "  subset=\"training\",\n",
    "  seed=123,\n",
    "  image_size=(img_height, img_width),\n",
    "  batch_size=batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_ds = tf.keras.preprocessing.image_dataset_from_directory(\n",
    "  val_dir,\n",
    "  validation_split=0.2,\n",
    "  subset=\"validation\",\n",
    "  seed=123,\n",
    "  image_size=(img_height, img_width),\n",
    "  batch_size=batch_size)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_classes = len(train_ds.class_names)\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "  layers.experimental.preprocessing.Rescaling(1./255),\n",
    "  layers.Conv2D(32, 3, activation='relu'),\n",
    "  layers.MaxPooling2D(),\n",
    "  layers.Conv2D(32, 3, activation='relu'),\n",
    "  layers.MaxPooling2D(),\n",
    "  layers.Conv2D(32, 3, activation='relu'),\n",
    "  layers.MaxPooling2D(),\n",
    "  layers.Flatten(),\n",
    "  layers.Dense(128, activation='relu'),\n",
    "  layers.Dense(num_classes)\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(\n",
    "  optimizer='adam',\n",
    "  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),\n",
    "  metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.fit(\n",
    "  train_ds,\n",
    "  validation_data=val_ds,\n",
    "  epochs=1\n",
    ")\n"
   ]
  }
 ]
}