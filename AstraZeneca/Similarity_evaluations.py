from scipy import ndimage
from google.colab import drive
drive.mount('/content/drive')
import nibabel as nib
import numpy as np
import os
import sys
import cv2
from skimage.transform import resize
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


def calculate_similarity(img_1,img_2):
  '''
  Enter image 1 and image 2 to calcualte mean square error and structural similarity.
  PSNR and SSIM are not used as they are more semantic metrics. Only MSE is used from this function.
  '''
  mse_1 = mean_squared_error(img_1,img_2)
  ssim_1 = ssim (img_1, img_2 ) # data_range=img_2.max() - img_2.min()) 
  psnr=cv2.PSNR(final,bin_mask_1)
  #print("PSNR:", round(psnr,4))
  print("MSE:", round(mse_1,5))
  #print("SSIM:", round(ssim_1,4))



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

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def dice_coef(img, img2):
        if img.shape != img2.shape:
            raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
        else:
            
            lenIntersection=0
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if ( np.array_equal(img[i][j],img2[i][j]) ):
                        lenIntersection+=1
             
            lenimg=img.shape[0]*img.shape[1]
            lenimg2=img2.shape[0]*img2.shape[1]  
            value = (2. * lenIntersection  / (lenimg + lenimg2))
        return value


MASK_files=os.listdir("/content/drive/MyDrive/AstraZeneca/SIM/MASK")
CAM_files=os.listdir("/content/drive/MyDrive/AstraZeneca/SIM/CAM")
MASK_files.sort()
CAM_files.sort()

for i in range(0,len(CAM_files)):
  MASK_img_path="/content/drive/MyDrive/AstraZeneca/SIM/MASK/" + MASK_files[i]
  CAM_img_path="/content/drive/MyDrive/AstraZeneca/SIM/CAM/" + CAM_files[i]
  img_1=read_nifti_file(MASK_img_path)
  mask_1=resize(img_1,(128,128,64))
  mask_1= ndimage.rotate(mask_1, 0, reshape=False)
  bin_mask_1= 1.0 * (mask_1 > 0.6667) #binarize original mask
  #msq = nib.Nifti1Image(mask_1, affine=None)
  #nib.save(msq, os.path.join('/content/drive/MyDrive/AstraZeneca/DICE', 'bin_mask_final' + MASK_files[i] ))
  cam_1=read_nifti_file(CAM_img_path)
  cam_1= ndimage.rotate(cam_1, 90, reshape=False) #rotate to right to fit overlay - this can be done manually 
  bin_cam_1 = 1.0 * (cam_1 > 0.7) #tresholds can be adjusted 
  #bin_cam_1 = 1.0 * (cam_1 > 0.1)
  addition= bin_mask_1 + bin_cam_1 # adds the overlap of the two masks
  final = 1.0 * (addition >= 2 ) #saves the intersection of the two masks
  addition_save = nib.Nifti1Image(final, affine=None)
  nib.save(addition_save, os.path.join('/content/drive/MyDrive/AstraZeneca/DICE', 'Addition_final_' + MASK_files[i] ))
  print("Similarity of:", MASK_files[i])
  calculate_similarity(final,bin_mask_1)
  print("Identical overlap", np.all(bin_mask_1 == final))# Check if the overlap fully (this happens if a low treshold is chosen)
  dice_score= dice_coef(final,bin_mask_1)
  print("DICE:" , round(dice_score,5))