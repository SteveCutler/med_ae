import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os


from utils import load_images, make_splits, ssim_mae_loss, get_latents

normal_dir = os.path.join(os.getcwd(), 'chest_xray/NORMAL')
pneu_dir = os.path.join(os.getcwd(), 'chest_xray/PNEUMONIA')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)


## LOAD IMAGES
normal_images, pneu_images = load_images(normal_dir, pneu_dir)

## PROCESS AND SPLIT IMAGES
val_x, val_y, test_x, test_y, norm_train, norm_val = make_splits(normal_images, pneu_images)

## load saved model
autoencoder = tf.keras.models.load_model("autoencoder_model_ssim.keras", custom_objects={"ssim_mae_loss": ssim_mae_loss})

## Grab the encoder feature map
enc_feat = autoencoder.layers[1].output

## Flatten it to a vector representation
enc_vec  = tf.keras.layers.GlobalAveragePooling2D()(enc_feat)
encoder_vec = tf.keras.Model(inputs=autoencoder.input, outputs=enc_vec)

print("encoder retrieved")

## Get latent space representations for normal images
latent_norm = get_latents(encoder_vec, norm_val)

## Get latent space representations for val set
latent_val = get_latents(encoder_vec, val_x)

print("latent normals & norm vals retrieved")

## Compute centroid of embedded normals and covariance
centroid = latent_norm.mean(axis=0)
print("centroid  retrieved")

cov = np.cov(latent_norm.T)   # covariance matrix
print("cov  retrieved")
cov_inv = np.linalg.pinv(cov) # pseudo-inverse for stability
print("inv cov  retrieved")



## Calculate distance from mean
scores_euclid = np.linalg.norm(latent_val - centroid, axis=1)

## Calculate Mahalanobis distance for comparison
def mahalanobis(x, mean, cov_inv):
    v = x - mean
    return np.sqrt(v @ cov_inv @ v.T)
scores_mahal = np.array([mahalanobis(z, centroid, cov_inv) for z in latent_val])



## run inference on validation/test
recons_val = autoencoder.predict(val_x)
errors_val = np.mean(np.abs(val_x - recons_val), axis=(1,2,3)) ## changed from MSE to MAE


## Inference on the test_x set
recons_test = autoencoder.predict(test_x)
errors_test = np.mean(np.abs(test_x - recons_test), axis=(1,2,3))  # MAE



## Compute SSIM for each sample in the val_x set
ssim_vals = []
for i in range(len(val_x)):
    ssim_val = tf.image.ssim(
        val_x[i:i+1], recons_val[i:i+1], max_val=1.0
    ).numpy()[0]
    ssim_vals.append(ssim_val)
ssim_vals = np.array(ssim_vals)

## Compute SSIM for each test sample in the test_x set
ssim_test = []
for i in range(len(test_x)):
    val = tf.image.ssim(test_x[i:i+1], recons_test[i:i+1], max_val=1.0).numpy()[0]
    ssim_test.append(val)
ssim_test = np.array(ssim_test)

## Hybrid score for val
scores_val = 0.5 * errors_val + 0.5 * (1 - ssim_vals)
scores_val = -scores_val

## Hybrid score for test
scores_test = 0.5 * errors_test + 0.5 * (1 - ssim_test)
scores_test = -scores_test


## Latent representations for test set
latent_test = get_latents(encoder_vec, test_x)

## Euclidean
scores_euclid_test = np.linalg.norm(latent_test - centroid, axis=1)

## Mahalanobis
scores_mahal_test = np.array([mahalanobis(z, centroid, cov_inv) for z in latent_test])



## Visualizing error distribution

# plt.hist(scores_val[val_y==0], bins=50, alpha=0.6, label="Normal", color ='blue')
# plt.hist(scores_val[val_y==1], bins=50, alpha=0.6, label="Pneumonia", color='red')
# plt.xlabel("Reconstruction error")
# plt.ylabel("Count")
# plt.legend()
# plt.title("Validation Reconstruction Errors")
# plt.show()


# # ROC curve
# fpr, tpr, thresholds = roc_curve(val_y, scores_val)
# roc_auc = auc(fpr, tpr)

# plt.plot(fpr, tpr, label=f"SSIM+MAE AUC = {roc_auc:.2f}")
# plt.plot([0,1],[0,1],'--',color='gray')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve (Validation)")
# plt.legend()
# plt.show()

## new ROC curves for comparison
fpr_e, tpr_e, _ = roc_curve(val_y, scores_euclid)
auc_e = auc(fpr_e, tpr_e)

fpr_m, tpr_m, _ = roc_curve(val_y, scores_mahal)
auc_m = auc(fpr_m, tpr_m)

## Plot comparison
plt.plot(fpr_e, tpr_e, label=f"Latent Euclidean AUC = {auc_e:.2f}")
plt.plot(fpr_m, tpr_m, label=f"Latent Mahalanobis AUC = {auc_m:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Latent Space Scoring)")
plt.legend()
plt.show()


# SSIM+MAE
fpr_h, tpr_h, _ = roc_curve(test_y, scores_test)
auc_h = auc(fpr_h, tpr_h)

# Euclidean
fpr_e, tpr_e, _ = roc_curve(test_y, scores_euclid_test)
auc_e = auc(fpr_e, tpr_e)

# Mahalanobis
fpr_m, tpr_m, _ = roc_curve(test_y, scores_mahal_test)
auc_m = auc(fpr_m, tpr_m)

# Plot all in one figure
plt.figure(figsize=(8,6))
plt.plot(fpr_h, tpr_h, label=f"SSIM+MAE AUC = {auc_h:.2f}")
plt.plot(fpr_e, tpr_e, label=f"Latent Euclidean AUC = {auc_e:.2f}")
plt.plot(fpr_m, tpr_m, label=f"Latent Mahalanobis AUC = {auc_m:.2f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend()
plt.show()
