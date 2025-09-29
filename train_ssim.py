import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model


from utils import load_images, make_splits, ssim_mae_loss

normal_dir = os.path.join(os.getcwd(), 'chest_xray/NORMAL')
pneu_dir = os.path.join(os.getcwd(), 'chest_xray/PNEUMONIA')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)


## LOAD IMAGES
normal_images, pneu_images = load_images(normal_dir, pneu_dir)

## PROCESS AND SPLIT IMAGES
val_x, val_y, test_x, test_y, norm_train, norm_val = make_splits(normal_images, pneu_images)


## Adding noise to the images
input_shape = (160, 160, 1)

noise_factor = 0.05
norm_noisy = norm_train + noise_factor * tf.random.normal(shape=norm_train.shape)
norm_val_noisy = norm_val + noise_factor * tf.random.normal(shape=norm_val.shape)

norm_noisy = tf.clip_by_value(norm_noisy, clip_value_min=0., clip_value_max=1.)
norm_val_noisy = tf.clip_by_value(norm_val_noisy, clip_value_min=0., clip_value_max=1.)

## Display noise images
# n = 10
# plt.figure(figsize=(20, 2))
# for i in range(n):
#     ax = plt.subplot(1, n, i + 1)
#     plt.title("original + noise")
#     plt.imshow(tf.squeeze(norm_noisy[i]))
#     plt.gray()
# plt.show()


## Creating Autoencoder Framework

## Encoder

encoder = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), padding='same'),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2), padding='same'),
    # layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    # layers.MaxPooling2D((2,2), padding='same'),
])


## Decoder

decoder = models.Sequential([
    # layers.Conv2DTranspose(256, (3,3), strides=2, activation = 'relu', padding = 'same'),
    layers.Conv2DTranspose(128, (3,3), strides=2, activation = 'relu', padding = 'same'),
    layers.Conv2DTranspose(64, (3,3), strides=2, activation = 'relu', padding = 'same'),
    layers.Conv2DTranspose(32, (3,3), strides=2, activation = 'relu', padding = 'same'),
    layers.Conv2D(1, (3,3), activation ='sigmoid', padding='same')
])

## Autoencoder

inputs = layers.Input(shape=input_shape)
encoded = encoder(inputs)
decoded = decoder(encoded)
autoencoder = models.Model(inputs, decoded)

## Use new MAE+SSIM loss function


autoencoder.compile(optimizer='adam', loss=ssim_mae_loss)
autoencoder.summary()


## Training the Autoencoder

EPOCHS=20

history = autoencoder.fit(
norm_noisy, norm_train,
epochs=EPOCHS,
batch_size=BATCH_SIZE,
validation_data=(norm_val_noisy, norm_val),
callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

reconstructions = autoencoder.predict(val_x)
errors = np.mean((val_x - reconstructions) ** 2, axis=(1,2,3))


## Saving the weights in TF format, as a folder:

autoencoder.save("autoencoder_model_ssim.keras")  

