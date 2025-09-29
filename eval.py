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

# Pick some samples from validation
n = 10  # number of images to display
samples = val_x[:n]

# Get reconstructions
recons = autoencoder.predict(samples)

plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(samples[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recons[i].squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()



## Visualizing error distribution

plt.hist(errors_val[val_y==0], bins=50, alpha=0.6, label="Normal", color ='blue')
plt.hist(errors_val[val_y==1], bins=50, alpha=0.6, label="Pneumonia", color='red')
plt.xlabel("Reconstruction error")
plt.ylabel("Count")
plt.legend()
plt.title("Validation Reconstruction Errors")
plt.show()


## Measure if able to determine pneu vs norm xrays

fpr, tpr, thresholds = roc_curve(val_y, errors_val)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend()
plt.show()
