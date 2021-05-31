import os
import zipfile
import numpy as np
import keras
import tensorflow as tf
import nibabel as nib
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from __future__ import absolute_import, division, print_function, unicode_literals
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.append('/content/drive/MyDrive/AstraZeneca')
from scipy import ndimage
import pandas as pd
from tensorflow.keras.optimizers import Adam
import cv2
from matplotlib import pyplot as plt


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


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


def get_last_conv_layer(keras_model):
  """ 
  Get last convolution layer name of keras model 
  input: Keras model 
  output: string of the name of the last convolution layer
  Make sure last convolution layer has "conv" in the name!
  """
  layer_names=[layer.name for layer in keras_model.layers]
  layer_names.reverse() 
  # This loop cycles the reversed list an extracts the name of the last conv layer
  for i in range(0,len(layer_names)):
    if "conv" in layer_names[i]:
      conv_layer=layer_names[i]; 
      print("The last convolution layer is:", conv_layer)
      return conv_layer


# GRAD CAM 3D


#Load the model and model weights

INPUT_PATCH_SIZE=(128,128,64,1) # same as the input in the model
inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
from keras.models import load_model
model = keras.models.load_model('/content/drive/My Drive/AstraZeneca/3D-11.h5')

Model_3D=model
#Model_3D.summary()
#Getting the last conv layer from the model using the function above
LAYER_NAME=get_last_conv_layer(Model_3D)

img_path="/content/drive/MyDrive/AstraZeneca/CT23/study_1100.nii"
# # Loading image with the nifti package and the preprocessing fucntions required for 3D scans
img=read_nifti_file(img_path)
n_img=normalize(img)
resized_img=resize_volume(n_img)
#try a prediction, trained model is from: https://keras.io/examples/vision/3D_image_classification/

prediction = Model_3D.predict(np.expand_dims(resized_img, axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )

# create a model that maps the input image to the activations
# of the last conv layer as well as the output predictions
# Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model([Model_3D.inputs], [Model_3D.get_layer(LAYER_NAME).output, Model_3D.output])
#grad_model.summary()

grad_model.layers[-1].activation = keras.activations.linear # This is done in order to get gradients and not softmax output (code works without it too?)

#Dimension are added to transform our array into a "batch" (required practice for keras/tensorflow)
#Expansion of first and last axis
io_img=tf.expand_dims(resized_img, axis=-1)
#print(io_img.shape)
io_img=tf.expand_dims(io_img, axis=0)
#print(io_img.shape)


# Class index is none because its softmax in this case
# This makes sense if the models uses categorical crossentropy and has more classes
CLASS_INDEX=None

# This Gradient tape function should be adapted depending on whether there is a 
# softmax or a categorical crossentropy loss
# Computing the gradient of the top predicted class for our input image
# with respect to the activations of the last conv layer
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(io_img)
    if len(predictions)==1:
      # Binary Classification
      loss = 1- predictions[0]  # This is what makes the scan "ABNORMAL"
      #loss = predictions[:, CLASS_INDEX] #This is the "anti class or what makes the scan "NORMAL"
    else:
      if CLASS_INDEX is None: # This is if there are more categories 
        CLASS_INDEX=tf.argmax(predictions[0]) #take top prediciton ?
        loss = predictions[:, CLASS_INDEX]
    

# This is the gradient of the output neuron (top predicted or chosen)
# with regard to the output feature map of the last conv layer    
# Extract filters and gradients of the top predicted class
output = conv_outputs[0] # output of the conv layer
grads = tape.gradient(loss, conv_outputs)[0]


# Guided gradient, to better visualize heatmap and offer more representable results, works without the next 3 lines but heatmaps look slightly different
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads


# Average gradients spatially
# This is a vector where each entry is the mean intensity of the gradient
# over a specific feature map channel
weights = tf.reduce_mean(guided_grads, axis=(0, 1,2))

# Build a wighted map of filters according to gradients importance
# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the top predicted class
# then sum all the channels to obtain the heatmap class activation
cam = np.zeros(output.shape[0:3], dtype=np.float32)
for index, w in enumerate(weights):
    cam = cam + w * output[:, :, :, index]

from skimage.transform import resize
from matplotlib import pyplot as plt

capi=resize(cam,(128,128,64)) #make the gradient cam the same size as the input
capi = np.maximum(capi,0)
heatmap = (capi - capi.min()) / (capi.max() - capi.min()) #normalize values between 0 and 1

#----------------------------------------------------------------
# Everything below is just plotting
# There are 3 viewpoint:  axial, coronal, saggital
# The scans are 3D: slices and viewpoints can be chosen
#----------------------------------------------------------------

f, axarr = plt.subplots(2,3,figsize=(15,10));
f.suptitle('Grad-CAM')
slice_count=-10    # slice range from -64 to 64 (depth)
slice_count2=39 #39-41


# Uncomment for Coronal viewpoint 
'''
ct_img_1=np.squeeze(resized_img[slice_count, :,:])
grad_cmap_img1=np.squeeze(heatmap[slice_count,:, :])

ct_img_2=np.squeeze(resized_img[:,slice_count2,:])
grad_cmap_img2=np.squeeze(heatmap[:,slice_count2,:]) 
'''
# Uncomment for Axial viewpoint 
ct_img_1=np.squeeze(resized_img[:, :,slice_count])
grad_cmap_img1=np.squeeze(heatmap[:, :,slice_count])

ct_img_2=np.squeeze(resized_img[:, :,slice_count2])
grad_cmap_img2=np.squeeze(heatmap[:, :,slice_count2]) 

# Uncomment for Saggital viewpoint 
'''
ct_img_1=np.squeeze(resized_img[:,slice_count,:])
grad_cmap_img1=np.squeeze(heatmap[:,slice_count, :])

ct_img_2=np.squeeze(resized_img[:,slice_count2,:])
grad_cmap_img2=np.squeeze(heatmap[:,slice_count2,:]) 
'''

# First slice

img_plot = axarr[0,0].imshow(ct_img_1, cmap='gray');
axarr[0,0].axis('off')
axarr[0,0].set_title('CT')
    
img_plot = axarr[0,1].imshow(grad_cmap_img1, cmap='jet');
axarr[0,1].axis('off')
axarr[0,1].set_title('Grad-CAM')
    
plot1_overlay=cv2.addWeighted(ct_img_1,0.4,grad_cmap_img1 ,0.6, 0) # Does an image overlay
    
img_plot = axarr[0,2].imshow(plot1_overlay,cmap='jet');
axarr[0,2].axis('off')
axarr[0,2].set_title('Overlay')

# Second slice

img_plot = axarr[1,0].imshow(ct_img_2, cmap='gray');
axarr[1,0].axis('off')
axarr[1,0].set_title('CT')
    
img_plot = axarr[1,1].imshow(grad_cmap_img2, cmap='jet');
axarr[1,1].axis('off')
axarr[1,1].set_title('Grad-CAM')
    
plot2_overlay=cv2.addWeighted(ct_img_2,0.4,grad_cmap_img2, 0.6, 0) #Does an image overlay
    
img_plot = axarr[1,2].imshow(plot2_overlay,cmap='jet');
axarr[1,2].axis('off')
axarr[1,2].set_title('Overlay')