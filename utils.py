import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

## Glob variables
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
input_shape = (160, 160, 1)


def load_images(normal_dir, pneu_dir):
    ## Load data from file
    train_dataset = tf.keras.utils.image_dataset_from_directory(normal_dir,
                                                                labels=None,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                color_mode="grayscale"
                                                                )

    val_dataset = tf.keras.utils.image_dataset_from_directory(pneu_dir,
                                                                labels=None,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                color_mode="grayscale"
                                                                )


    ## Display images

    # plt.figure(figsize=(10, 10))
    # for images in train_dataset.take(1):
    #   for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(images[i].numpy().astype("uint8").squeeze(), cmap="gray")
    #     plt.axis("off")
    # plt.show()    


    ## Convert to Numpy arrays

    normal_images = np.concatenate([x.numpy() for x in train_dataset], axis=0)
    pneu_images = np.concatenate([x.numpy() for x in val_dataset], axis=0)


    ## Normalize the image pixel values

    normal_images = normal_images.astype('float32')/255
    pneu_images = pneu_images.astype('float32')/255

    return normal_images, pneu_images



def make_splits(normal_images, pneu_images):
    ## Take 30% of normal images and split them up into val_set and test_set

    norm_train, valtest_set = train_test_split(normal_images, test_size=0.3, random_state=42)
    norm_val, norm_test = train_test_split(valtest_set, test_size=0.5, random_state=42)

    ## Split up pneu images into val and test sets

    pneu_val, pneu_test = train_test_split(pneu_images,test_size=0.5,random_state=42)

    ## SET BREAKDOWN
    ## norm_train used for just training the model

    ## norm_val used in training and threshold calibration
    ## pneu_val used in threshold calibration

    ## norm_test and pneu test used in final testing

    ## so we need:
    #  a calibration set: 237 norm val, 237 pneu val
    #  a test set: 238 norm_test, 238 val_test



    ## Culling unnecessary Pneu images to equalize set lengths before combining

    pneu_val = pneu_val[:len(norm_val)]
    pneu_test = pneu_test[:len(norm_test)]

    # print("norm_train set length: ", len(norm_train))
    # print("norm_val set length: ", len(norm_val))
    # print("norm_test set length: ", len(norm_test))
    # print("pneu_val set length: ", len(pneu_val))
    # print("pneu_test set length: ", len(pneu_test))


    ## Combining the Val sets and Test sets and creating the label arrays

    val_x = np.concatenate([norm_val, pneu_val])
    val_y = np.concatenate([np.zeros(len(norm_val)), np.ones(len(pneu_val))])

    test_x = np.concatenate([norm_test,pneu_test])
    test_y = np.concatenate([np.zeros(len(norm_test)),np.ones(len(pneu_test))])

    return val_x, val_y, test_x, test_y, norm_train, norm_val

# Define new loss paradigm: SSIM+MAE 
def ssim_mae_loss(y_true, y_pred):
    # Mean Absolute Error
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    # SSIM similarity (closer to 1 means better)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    # Hybrid: lower = better
    return 0.5 * mae + (1 - 0.5) * (1 - ssim)

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

## latent representation helper function
def get_latents(enc, data, batch_size=32):
    latents = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        z = enc.predict(data[i:i+batch_size], verbose=0)
        latents.append(z)
    return np.concatenate(latents, axis=0)
