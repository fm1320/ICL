from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import ndimage
import os
import zipfile
import numpy as np
import tensorflow as tf
import nibabel as nib
import datetime
import sys

NII_FILE = "study.nii"
MODEL_H5 = '3d.h5'
CONV_LAYER = 'conv3d_3'

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

INPUT_PATCH_SIZE=(128,128,64,1) # same as the input in the model
inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
model = load_model(MODEL_H5)
print(model.summary())

nii_scan = nib.load(NII_FILE)
img = nii_scan.get_fdata()
n_img=normalize(img)
resized_img=resize_volume(n_img)

prediction = model.predict(np.expand_dims(resized_img, axis=0))[0]
print("This model is %.2f percent confident that CT scan is abnormal" \
    % ((100 * prediction[0]), ) \
)

grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(CONV_LAYER).output, model.output])

with tf.GradientTape() as tape:
    conv_out, preds = grad_model(np.expand_dims(resized_img, axis=0))
    channel = preds[:, 0]

grads = tape.gradient(channel, conv_out)

print("conv_out shape", conv_out.shape)
print("grads shape", grads.shape)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

print("pooled_grads shape", pooled_grads.shape)

conv_out = conv_out[0]
heatmap = conv_out @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)
heatmap = tf.reduce_mean(heatmap, axis=(2,))
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
print("heatmap shape", heatmap.shape)

# Rotate img, make grayscale img and heatmap
the_img = ndimage.rotate(n_img, 90, reshape=False)
img = np.uint8(255 * the_img[:,:,10])
heatmap = np.uint8(255 * heatmap)

gray_colors = cm.get_cmap("gray")(np.arange(256))[:,:3]
img = gray_colors[img]

# Use jet blue for heatmap
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:,:3]
jet_heatmap = jet_colors[heatmap]

# Resize heatmap to fit image
jet_heatmap = array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = img_to_array(jet_heatmap)

fig, axs = plt.subplots(1, 2)
old_img = array_to_img(img)
full_img = array_to_img(img + 0.05 * jet_heatmap)
axs[0].imshow(old_img)
axs[1].imshow(full_img)
plt.show()