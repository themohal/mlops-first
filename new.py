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
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "17747\nThere was an error.\n5365\nThere was an error.\n0\n"
     ]
    }
   ],
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
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Found 17783 files belonging to 34 classes.\nUsing 14227 files for training.\n"
     ]
    }
   ],
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
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Found 5492 files belonging to 34 classes.\nUsing 1098 files for validation.\n"
     ]
    }
   ],
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
   "execution_count": 28,
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
   "execution_count": 29,
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
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "Epoch 1/3\n",
      " 62/445 [===>..........................] - ETA: 14:03 - loss: 3.3260 - accuracy: 0.0796"
     ]
    },
    {
     "output_type": "error",
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-30-ce06c3301011>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m model.fit(\n\u001b[0m\u001b[0;32m      2\u001b[0m   \u001b[0mtrain_ds\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m   \u001b[0mvalidation_data\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mval_ds\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m   \u001b[0mepochs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m3\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m )\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\keras\\engine\\training.py\u001b[0m in \u001b[0;36m_method_wrapper\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m    106\u001b[0m   \u001b[1;32mdef\u001b[0m \u001b[0m_method_wrapper\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    107\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_in_multi_worker_mode\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m  \u001b[1;31m# pylint: disable=protected-access\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 108\u001b[1;33m       \u001b[1;32mreturn\u001b[0m \u001b[0mmethod\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    109\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    110\u001b[0m     \u001b[1;31m# Running inside `run_distribute_coordinator` already.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)\u001b[0m\n\u001b[0;32m   1096\u001b[0m                 batch_size=batch_size):\n\u001b[0;32m   1097\u001b[0m               \u001b[0mcallbacks\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mon_train_batch_begin\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mstep\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1098\u001b[1;33m               \u001b[0mtmp_logs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtrain_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1099\u001b[0m               \u001b[1;32mif\u001b[0m \u001b[0mdata_handler\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshould_sync\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1100\u001b[0m                 \u001b[0mcontext\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masync_wait\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\eager\\def_function.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwds)\u001b[0m\n\u001b[0;32m    778\u001b[0m       \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    779\u001b[0m         \u001b[0mcompiler\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m\"nonXla\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 780\u001b[1;33m         \u001b[0mresult\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    781\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    782\u001b[0m       \u001b[0mnew_tracing_count\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_tracing_count\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\eager\\def_function.py\u001b[0m in \u001b[0;36m_call\u001b[1;34m(self, *args, **kwds)\u001b[0m\n\u001b[0;32m    805\u001b[0m       \u001b[1;31m# In this case we have created variables on the first call, so we run the\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    806\u001b[0m       \u001b[1;31m# defunned version which is guaranteed to never create variables.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 807\u001b[1;33m       \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_stateless_fn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m  \u001b[1;31m# pylint: disable=not-callable\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    808\u001b[0m     \u001b[1;32melif\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_stateful_fn\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    809\u001b[0m       \u001b[1;31m# Release the lock early so that multiple threads can perform the call\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\eager\\function.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   2827\u001b[0m     \u001b[1;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2828\u001b[0m       \u001b[0mgraph_function\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_maybe_define_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2829\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0mgraph_function\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_filtered_call\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m  \u001b[1;31m# pylint: disable=protected-access\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2830\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2831\u001b[0m   \u001b[1;33m@\u001b[0m\u001b[0mproperty\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_filtered_call\u001b[1;34m(self, args, kwargs, cancellation_manager)\u001b[0m\n\u001b[0;32m   1841\u001b[0m       \u001b[0;31m`\u001b[0m\u001b[0margs\u001b[0m\u001b[0;31m`\u001b[0m \u001b[1;32mand\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m`\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;31m`\u001b[0m\u001b[1;33m.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1842\u001b[0m     \"\"\"\n\u001b[1;32m-> 1843\u001b[1;33m     return self._call_flat(\n\u001b[0m\u001b[0;32m   1844\u001b[0m         [t for t in nest.flatten((args, kwargs), expand_composites=True)\n\u001b[0;32m   1845\u001b[0m          if isinstance(t, (ops.Tensor,\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\eager\\function.py\u001b[0m in \u001b[0;36m_call_flat\u001b[1;34m(self, args, captured_inputs, cancellation_manager)\u001b[0m\n\u001b[0;32m   1921\u001b[0m         and executing_eagerly):\n\u001b[0;32m   1922\u001b[0m       \u001b[1;31m# No tape is watching; skip to running the function.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1923\u001b[1;33m       return self._build_call_outputs(self._inference_function.call(\n\u001b[0m\u001b[0;32m   1924\u001b[0m           ctx, args, cancellation_manager=cancellation_manager))\n\u001b[0;32m   1925\u001b[0m     forward_backward = self._select_forward_and_backward_functions(\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\eager\\function.py\u001b[0m in \u001b[0;36mcall\u001b[1;34m(self, ctx, args, cancellation_manager)\u001b[0m\n\u001b[0;32m    543\u001b[0m       \u001b[1;32mwith\u001b[0m \u001b[0m_InterpolateFunctionError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    544\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mcancellation_manager\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 545\u001b[1;33m           outputs = execute.execute(\n\u001b[0m\u001b[0;32m    546\u001b[0m               \u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msignature\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    547\u001b[0m               \u001b[0mnum_outputs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_num_outputs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\envs\\tensorflow\\lib\\site-packages\\tensorflow\\python\\eager\\execute.py\u001b[0m in \u001b[0;36mquick_execute\u001b[1;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[0;32m     57\u001b[0m   \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     58\u001b[0m     \u001b[0mctx\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mensure_initialized\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 59\u001b[1;33m     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,\n\u001b[0m\u001b[0;32m     60\u001b[0m                                         inputs, attrs, num_outputs)\n\u001b[0;32m     61\u001b[0m   \u001b[1;32mexcept\u001b[0m \u001b[0mcore\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_NotOkStatusException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "model.fit(\n",
    "  train_ds,\n",
    "  validation_data=val_ds,\n",
    "  epochs=3\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ]
}
