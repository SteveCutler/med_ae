import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os


from utils import load_images, make_splits, ssim_mae_loss

normal_dir = os.path.join(os.getcwd(), 'chest_xray/NORMAL')
pneu_dir = os.path.join(os.getcwd(), 'chest_xray/PNEUMONIA')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)


## LOAD IMAGES
normal_images, pneu_images = load_images(normal_dir, pneu_dir)

## PROCESS AND SPLIT IMAGES
val_x, val_y, test_x, test_y, norm_train, norm_val = make_splits(normal_images, pneu_images)

# Adding noise
# noise_factor = 0.05
# val_x = val_x + noise_factor * tf.random.normal(shape=val_x.shape)
# val_x = tf.clip_by_value(val_x, clip_value_min=0., clip_value_max=1.)


# load saved modelx
autoencoder = tf.keras.models.load_model("autoencoder_model_ssim.keras", custom_objects={"ssim_mae_loss": ssim_mae_loss})

# run inference on validation/test
recons_val = autoencoder.predict(val_x)
errors_val = np.mean(np.abs(val_x - recons_val), axis=(1,2,3)) ## changed from MSE to MAE


# # Pick some samples from validation
# n = 10  # number of images to display
# samples = val_x[:n]
# # Get reconstructions
# recons = autoencoder.predict(samples)

# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # Original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(samples[i].squeeze(), cmap="gray")
#     plt.title("Original")
#     plt.axis("off")

#     # Reconstructed
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(recons[i].squeeze(), cmap="gray")
#     plt.title("Reconstructed")
#     plt.axis("off")

# plt.show()

## Compute SSIM for each sample
ssim_vals = []
for i in range(len(val_x)):
    ssim_val = tf.image.ssim(
        val_x[i:i+1], recons_val[i:i+1], max_val=1.0
    ).numpy()[0]
    ssim_vals.append(ssim_val)
ssim_vals = np.array(ssim_vals)

## Hybrid score (higher = more anomalous)
scores_val = 0.5 * errors_val + 0.5 * (1 - ssim_vals)



## Visualizing error distribution

plt.hist(scores_val[val_y==0], bins=50, alpha=0.6, label="Normal", color ='blue')
plt.hist(scores_val[val_y==1], bins=50, alpha=0.6, label="Pneumonia", color='red')
plt.xlabel("Reconstruction error")
plt.ylabel("Count")
plt.legend()
plt.title("Validation Reconstruction Errors")
plt.show()


# ROC curve
fpr, tpr, thresholds = roc_curve(val_y, scores_val)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"SSIM+MAE AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation)")
plt.legend()
plt.show()