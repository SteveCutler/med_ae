import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os


from utils import load_images, make_splits

normal_dir = os.path.join(os.getcwd(), 'chest_xray/NORMAL')
pneu_dir = os.path.join(os.getcwd(), 'chest_xray/PNEUMONIA')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)


## LOAD IMAGES
normal_images, pneu_images = load_images(normal_dir, pneu_dir)

## PROCESS AND SPLIT IMAGES
val_x, val_y, test_x, test_y, norm_train, norm_val = make_splits(normal_images, pneu_images)


# load saved model
autoencoder = tf.keras.models.load_model("autoencoder_model.keras")

# run inference on validation/test
recons_val = autoencoder.predict(val_x)
errors_val = np.mean((val_x - recons_val)**2, axis=(1,2,3))