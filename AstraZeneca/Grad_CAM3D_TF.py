from __future__ import absolute_import, division, print_function, unicode_literals
import os
import zipfile
import numpy as np
import tensorflow as tf
import nibabel as nib
import datetime
from tensorflow import keras
from tensorflow.keras import layers
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
    """Normalize the volume and define the Hausfeld units range"""
    min = -1000
    max = 500
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



#Load the model and model weights

input_size=(128,128,64,1) # same as the input in the model
inputs = tf.keras.Input(shape=input_size, name='CT')
from keras.models import load_model
model = keras.models.load_model('/content/drive/My Drive/AstraZeneca/3D-11.h5') # loading weights of trained model

Model_3D=model
#Model_3D.summary()  # Summary of the loaded model
#Getting the last conv layer from the model using the function above
layer_name=get_last_conv_layer(Model_3D)

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
grad_model = tf.keras.models.Model([Model_3D.inputs], [Model_3D.get_layer(layer_name).output, Model_3D.output])


grad_model.layers[-1].activation = keras.activations.linear # This is done in order to get linear activation at the last laeyr and not softmax output 

#Dimension are added to transform our array into a "batch" (required practice for keras/tensorflow)
#Expansion of first and last axis
io_img=tf.expand_dims(resized_img, axis=-1)
#print(io_img.shape)
io_img=tf.expand_dims(io_img, axis=0)
#print(io_img.shape)


# Class index is none because its softmax in this case
# This makes sense if the models uses categorical crossentropy and has more classes
class_idx=None
# - The code should work for other CNN based models both with binary and mutiple class classification
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
      if class_idx is None: # This is if there are more categories 
        class_idx=tf.argmax(predictions[0]) #take top prediciton 
        loss = predictions[:, class_idx]
    

# This is the gradient of the output neuron (top predicted or chosen)
# with regard to the output feature map of the last conv layer    
# Extract filters and gradients of the top predicted class
# [0] is used in order to "lose" the batch dimension
output = conv_outputs[0] # output of the conv layer
grads = tape.gradient(loss, conv_outputs)[0]


# Guided gradient, to better visualize heatmap and offer more representable results, works without the next 3 lines but heatmaps look slightly different
# It uses the idea of ReLU where negative values are supressed
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads  #Following the example of guided gradcam 


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

cam=resize(cam,(128,128,64)) #make the gradient cam the same size as the input
cam = np.maximum(cam,0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min()) #normalize values between 0 and 1

#----------------------------------------------------------------
# Everything below is plotting
# There are 3 viewpoint:  axial, coronal, saggital
# The scans are 3D: slices and viewpoints can be chosen
# Viridis colormap is chosen instead of Jet because of the better color coding 
# of smaller differences in values
#----------------------------------------------------------------


# plotting original CT scans 
def plot_slices_CT(num_rows, num_columns, width, height, data):
    """Plots 40 slices in their original """
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 80.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    f.suptitle('CT-scans')

    for i in range(rows_data):
        for j in range(columns_data):
            img_plot= axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
            f.colorbar(img_plot, ax=axarr[i,j])
    plt.show()
# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices_CT(4, 10, 128, 128, np.squeeze(resized_img[:, :,:40]))

# plotting heatmaps
def plot_slices_HM(num_rows, num_columns, width, height, data):
    """Plot heatmap of CAM slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 80.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    f.suptitle('Heatmaps')

    for i in range(rows_data):
        for j in range(columns_data):
            img_plot= axarr[i, j].imshow(data[i][j], cmap="viridis")
            axarr[i, j].axis("off")
            f.colorbar(img_plot, ax=axarr[i,j], ticks=np.arange(0.0, 1.0, 0.1))
    plt.show()
# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices_HM(4, 10, 128, 128, np.squeeze(heatmap[:, :,:40]))


def plot_slices_OL(num_rows, num_columns, width, height, CT_data, HM_data):
    """Plot overlays of slices"""
    CT_data = np.rot90(np.array(CT_data))
    CT_data = np.transpose(CT_data)
    CT_data = np.reshape(CT_data, (num_rows, num_columns, width, height))
    rows_data, columns_data = CT_data.shape[0], CT_data.shape[1]
    heights = [slc[0].shape[0] for slc in CT_data]
    widths = [slc.shape[1] for slc in CT_data[0]]

    HM_data = np.rot90(np.array(HM_data))
    HM_data= np.transpose(HM_data)
    HM_data= np.reshape(HM_data, (num_rows, num_columns, width, height))
    rows_data, columns_data = HM_data.shape[0], HM_data.shape[1]
    heights = [slc[0].shape[0] for slc in HM_data]
    widths = [slc.shape[1] for slc in HM_data[0]]


    fig_width = 80.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    f.suptitle('Overlay maps')

    for i in range(rows_data):
        for j in range(columns_data):
            plot_overlay=cv2.addWeighted(CT_data[i][j], 0.4, HM_data[i][j], 0.6, 0)
            img_plot= axarr[i, j].imshow(plot_overlay, cmap="viridis")
            axarr[i, j].axis("off")
            f.colorbar(img_plot, ax=axarr[i,j], ticks=np.arange(0.0, 1.0, 0.1) )
    plt.show()

plot_slices_OL(4, 10, 128, 128, np.squeeze(resized_img[:, :,:40]) , np.squeeze(heatmap[:, :,:40]) )


#####################################################################################################################
# The code below plots a given number of slice and it's heatmaps
# Also it's not only from an axial perspective but saggital and coronal can be chosen
# However it's impornant to check which slices are being plotted as usually diffrent viewpoints have different number of slices
#########################################################################################################################
f, axarr = plt.subplots(2,3,figsize=(18,12));
f.suptitle('Grad-CAM')
slice_count=25
slice_count2= 23
# Different perspectives have different number of slices
# In the written report the slices are:
# for axial: 25,23 
# for saggital: 98,100
# for coronal : 45,70 


# Uncomment for Coronal viewpoint 
'''
ct_img_1=np.rot90(np.squeeze(resized_img[slice_count, :,:]))
grad_cmap_img1=np.rot90(np.squeeze(heatmap[slice_count,:, :]))

ct_img_2=np.rot90(np.squeeze(resized_img[slice_count2,:,:]))
grad_cmap_img2=np.rot90(np.squeeze(heatmap[slice_count2,:,:])) 
'''

# Uncomment for Axial viewpoint 
ct_img_1=(np.squeeze(resized_img[:, :,slice_count]))
grad_cmap_img1=(np.squeeze(heatmap[:, :,slice_count]))

ct_img_2=(np.squeeze(resized_img[:, :,slice_count2]))
grad_cmap_img2=(np.squeeze(heatmap[:, :,slice_count2])) 


# Uncomment for Saggital viewpoint 
'''
ct_img_1=np.rot90(np.squeeze(resized_img[:,slice_count,:]))
grad_cmap_img1=np.rot90(np.squeeze(heatmap[:,slice_count, :]))

ct_img_2=np.rot90(np.squeeze(resized_img[:,slice_count2,:]))
grad_cmap_img2=np.rot90(np.squeeze(heatmap[:,slice_count2,:]))
'''

# First slice

img_plot = axarr[0,0].imshow(ct_img_1, cmap='gray');
axarr[0,0].axis('off')
axarr[0,0].set_title('CT')
f.colorbar(img_plot, ax=axarr[0,0], fraction=0.046, pad=0.05, ticks=np.arange(0.0, 1.0, 0.1)) # the ticks and fraction and pad parameters just 
                                                                                              # change and position the colorbar's apperance

    
img_plot = axarr[0,1].imshow(grad_cmap_img1, cmap='viridis');
axarr[0,1].axis('off')
axarr[0,1].set_title('Grad-CAM')
f.colorbar(img_plot, ax=axarr[0,1], fraction=0.046, pad=0.05, ticks=np.arange(0.0, 1.0, 0.1))

plot1_overlay=cv2.addWeighted(ct_img_1,0.4,grad_cmap_img1 ,0.6, 0) # Does an image overlay
    
img_plot = axarr[0,2].imshow(plot1_overlay,cmap='viridis');
axarr[0,2].axis('off')
axarr[0,2].set_title('Overlay')
f.colorbar(img_plot, ax=axarr[0,2], fraction=0.046, pad=0.05, ticks=np.arange(0.0, 1.0, 0.1))

# Second slice

img_plot = axarr[1,0].imshow(ct_img_2, cmap='gray');
axarr[1,0].axis('off')
axarr[1,0].set_title('CT')
f.colorbar(img_plot, ax=axarr[1,0], fraction=0.046, pad=0.05, ticks=np.arange(0.0, 1.0, 0.1))

img_plot = axarr[1,1].imshow(grad_cmap_img2, cmap='viridis');
axarr[1,1].axis('off')
axarr[1,1].set_title('Grad-CAM')
f.colorbar(img_plot, ax=axarr[1,1], fraction=0.046, pad=0.05, ticks=np.arange(0.0, 1.0, 0.1))

plot2_overlay=cv2.addWeighted(ct_img_2,0.4,grad_cmap_img2, 0.6, 0) #Does an image overlay

    
img_plot = axarr[1,2].imshow(plot2_overlay,cmap='viridis');
axarr[1,2].axis('off')
axarr[1,2].set_title('Overlay')
f.colorbar(img_plot, ax=axarr[1,2], fraction=0.046, pad=0.05, ticks=np.arange(0.0, 0.81, 0.1))