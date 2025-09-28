import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model


from utils import load_images, make_splits

normal_dir = os.path.join(os.getcwd(), 'chest_xray/NORMAL')
pneu_dir = os.path.join(os.getcwd(), 'chest_xray/PNEUMONIA')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)


## LOAD IMAGES
normal_images, pneu_images = load_images(normal_dir, pneu_dir)

## PROCESS AND SPLIT IMAGES
val_x, val_y, test_x, test_y, norm_train, norm_val = make_splits(normal_images, pneu_images)

## Creating Autoencoder Framework

input_shape = (160, 160, 1)

## Encoder

encoder = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), padding='same'),
])

## Decoder

decoder = models.Sequential([
    layers.Conv2DTranspose(64, (3,3), strides=2, activation = 'relu', padding = 'same'),
    layers.Conv2DTranspose(32, (3,3), strides=2, activation = 'relu', padding = 'same'),
    layers.Conv2D(1, (3,3), activation ='sigmoid', padding='same')
])

## Autoencoder

inputs = layers.Input(shape=input_shape)
encoded = encoder(inputs)
decoded = decoder(encoded)
autoencoder = models.Model(inputs, decoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


## Training the Autoencoder

EPOCHS=20

history = autoencoder.fit(
norm_train, norm_train,
epochs=EPOCHS,
batch_size=BATCH_SIZE,
validation_data=(norm_val, norm_val),
callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

reconstructions = autoencoder.predict(val_x)
errors = np.mean((val_x - reconstructions) ** 2, axis=(1,2,3))


## Saving the weights in TF format, as a folder:

autoencoder.save("autoencoder_model.keras")  

