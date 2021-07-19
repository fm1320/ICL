3D nifti files of the heatmaps. The heatmaps are casted to 16 bit precision by ITK snap.
Also the images are rotated by 90 degrees. (The bottom side of the scan is on the left).
The files which contain "resz" in their name are resized versions of the original. 
They were resized to 128x128x64 and these dimensions were used for the model training and 
heatmap generation. 
The files which containt "cam" in their name are the 3D heatmaps of the resized scans.
